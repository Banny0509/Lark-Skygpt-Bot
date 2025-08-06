import httpx
import os
import json
import logging
import re
from fastapi import FastAPI, Request, HTTPException

# --- 1. 日誌設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 2. 智慧載入環境變數 ---
if "RAILWAY_ENVIRONMENT" not in os.environ:
    from dotenv import load_dotenv
    logger.info("偵測到非生產環境，正在載入 .env 檔案...")
    load_dotenv()
else:
    logger.info("偵測到生產環境，將直接使用平台設定的環境變數。")

# --- 3. 初始化 FastAPI 應用 ---
app = FastAPI()

# --- 4. 讀取並驗證設定值 ---
LARK_APP_ID = os.getenv("APP_ID")
LARK_APP_SECRET = os.getenv("APP_SECRET")
VERIFICATION_TOKEN = os.getenv("VERIFICATION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([LARK_APP_ID, LARK_APP_SECRET, VERIFICATION_TOKEN, OPENAI_API_KEY]):
    logger.critical("一個或多個必要的環境變數未設定！")

# --- 5. 健康檢查端點 ---
@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(request: Request):
    payload_bytes = await request.body()
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError:
        logger.error("收到了無效的 JSON 格式請求。")
        raise HTTPException(status_code=400, detail="無效的 JSON 格式。")

    if "challenge" in payload:
        logger.info("收到 URL 驗證挑戰，已成功回應。")
        return {"challenge": payload["challenge"]}

    header = payload.get("header", {})
    if header.get("token") != VERIFICATION_TOKEN:
        logger.warning(f"收到了無效的 Token: {header.get('token')}")
        raise HTTPException(status_code=403, detail="無效的 Token。")

    event_type = header.get("event_type")
    if event_type == "im.message.receive_v1":
        try:
            await handle_message_receive(payload.get("event", {}))
        except Exception as e:
            logger.error(f"處理訊息時發生未預期錯誤: {e}", exc_info=True)
    else:
        logger.info(f"收到了暫不處理的事件類型: {event_type}")

    return {"code": 0}


async def handle_message_receive(event: dict):
    message = event.get("message", {})
    if not message:
        return

    message_type = message.get("message_type")
    chat_id = message.get("chat_id")
    
    if message_type != "text":
        logger.info(f"忽略非文字訊息 (類型: {message_type})。")
        return

    try:
        content_str = message.get("content", "{}")
        content_dict = json.loads(content_str)
        text_from_lark = content_dict.get("text", "")
    except Exception:
        logger.error(f"解析訊息 content 失敗, 原始 content: {message.get('content')}")
        return

    user_text = re.sub(r'<at.*?</at>', '', text_from_lark).strip()

    if not user_text:
        logger.info("移除 @提及 後訊息為空，已忽略。")
        return

    logger.info(f"從 chat_id {chat_id} 收到有效問題: '{user_text}'")

    try:
        chatgpt_reply = await get_chatgpt_response(user_text)
        await send_message_to_lark(chat_id, chatgpt_reply)
    except Exception as e:
        logger.error(f"與外部 API 互動時出錯: {e}", exc_info=True)
        await send_message_to_lark(chat_id, "抱歉，我現在遇到一點問題，請稍後再試。")


async def get_chatgpt_response(prompt: str) -> str:
    """
    向 OpenAI API 發送請求並獲取回應。
    """
    logger.info("--- 正在呼叫 OpenAI API ---")
    
    # --- 【關鍵修正】在日誌中安全地印出正在使用的金鑰片段以供驗證 ---
    if OPENAI_API_KEY and len(OPENAI_API_KEY) > 9:
        logger.info(f"正在使用的 OpenAI 金鑰片段: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}")
    else:
        logger.error("偵測到 OpenAI 金鑰不存在或格式不正確！")
        raise Exception("OpenAI 金鑰不存在或格式不正確。")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={"model": "gpt-4", "messages": [{"role": "user", "content": prompt}]},
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            reply_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not reply_text:
                logger.warning("OpenAI 返回了空的回應。")
                return "抱歉，我思考了一下，但沒有找到答案。"
            return reply_text
    except httpx.HTTPStatusError as e:
        logger.error(f"OpenAI API 請求失敗，狀態碼: {e.response.status_code}, 回應: {e.response.text}")
        raise Exception("OpenAI API 請求失敗。")
    except Exception as e:
        logger.error(f"呼叫 OpenAI API 時發生未預期錯誤: {e}", exc_info=True)
        raise e


async def get_lark_token() -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": LARK_APP_ID, "app_secret": LARK_APP_SECRET}
        )
        response.raise_for_status()
        data = response.json()
        token = data.get("tenant_access_token")
        if not token:
            raise Exception(f"獲取 Lark Token 失敗，回應: {data}")
        return token


async def send_message_to_lark(chat_id: str, text: str):
    try:
        token = await get_lark_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        payload = {
            "receive_id": chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": text})
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://open.larksuite.com/open-apis/im/v1/messages?receive_id_type=chat_id",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            if result.get("code") == 0:
                logger.info("訊息已成功發送到 Lark。")
            else:
                logger.error(f"發送 Lark 訊息失敗: Code={result.get('code')}, Msg={result.get('msg')}, RequestID: {response.headers.get('X-Request-Id')}")
    except Exception as e:
        logger.error(f"發送訊息到 Lark 時發生嚴重錯誤: {e}", exc_info=True)

# 強制觸發一次新的部署 v1.0

