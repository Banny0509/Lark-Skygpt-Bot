import httpx
import os
import json
import logging
import re  # 引入正則表達式模組
from fastapi import FastAPI, Request, HTTPException

# --- 1. 日誌設定 ---
# 這是您在伺服器上觀察機器人行為的眼睛，非常重要。
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 2. 智慧載入環境變數 ---
# 判斷是否在 Railway 等生產環境，如果是，就直接用平台變數，否則才讀取本地 .env
if "RAILWAY_ENVIRONMENT" not in os.environ:
    from dotenv import load_dotenv
    logger.info("偵測到非生產環境，正在載入 .env 檔案...")
    load_dotenv()
else:
    logger.info("偵測到生產環境，將直接使用平台設定的環境變數。")

# --- 3. 初始化 FastAPI 應用 ---
app = FastAPI()

# --- 4. 讀取設定值 ---
LARK_APP_ID = os.getenv("APP_ID")
LARK_APP_SECRET = os.getenv("APP_SECRET")
VERIFICATION_TOKEN = os.getenv("VERIFICATION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 啟動時檢查，確保所有金鑰都已設定
if not all([LARK_APP_ID, LARK_APP_SECRET, VERIFICATION_TOKEN, OPENAI_API_KEY]):
    logger.critical("一個或多個必要的環境變數未設定！應用程式無法啟動。")
    # 如果您希望在缺少金鑰時直接讓應用崩潰，可以取消下面這行的註解
    # raise ValueError("一個或多個必要的環境變數未設定！")

# --- 5. 健康檢查端點 ---
# 這是提供給 Railway 平台的「心跳」，告訴它「我還活著」，避免被誤殺。
@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(request: Request):
    """
    接收並處理來自 Lark 的所有 Webhook 事件。
    """
    payload_bytes = await request.body()
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError:
        logger.error("收到了無效的 JSON 格式請求。")
        raise HTTPException(status_code=400, detail="無效的 JSON 格式。")

    # 處理 Lark 的 URL 驗證挑戰
    if "challenge" in payload:
        logger.info("收到 URL 驗證挑戰，已成功回應。")
        return {"challenge": payload["challenge"]}

    # 驗證事件來源是否為我們的 Lark 應用
    header = payload.get("header", {})
    if header.get("token") != VERIFICATION_TOKEN:
        logger.warning(f"收到了無效的 Token: {header.get('token')}")
        raise HTTPException(status_code=403, detail="無效的 Token。")

    # 根據事件類型，交給對應的函式處理
    event_type = header.get("event_type")
    if event_type == "im.message.receive_v1":
        try:
            await handle_message_receive(payload.get("event", {}))
        except Exception as e:
            logger.error(f"處理訊息時發生未預期錯誤: {e}", exc_info=True)
    else:
        logger.info(f"收到了暫不處理的事件類型: {event_type}")

    # 向 Lark 回應成功，避免它重複發送同一個事件
    return {"code": 0}


async def handle_message_receive(event: dict):
    """
    專門處理「接收訊息」事件。
    """
    message = event.get("message", {})
    if not message:
        return

    message_type = message.get("message_type")
    chat_id = message.get("chat_id")
    
    # 我們只處理文字訊息
    if message_type != "text":
        logger.info(f"忽略非文字訊息 (類型: {message_type})。")
        return

    # 解析 Lark 傳來的 content JSON 字串
    try:
        content_str = message.get("content", "{}")
        content_dict = json.loads(content_str)
        text_from_lark = content_dict.get("text", "")
    except Exception:
        logger.error(f"解析訊息 content 失敗, 原始 content: {message.get('content')}")
        return

    # 使用正則表達式，穩定地移除所有 @提及 標籤
    user_text = re.sub(r'<at.*?</at>', '', text_from_lark).strip()

    # 如果只 @機器人 而沒有其他文字，則忽略
    if not user_text:
        logger.info("移除 @提及 後訊息為空，已忽略。")
        return

    logger.info(f"從 chat_id {chat_id} 收到有效問題: '{user_text}'")

    try:
        chatgpt_reply = await get_chatgpt_response(user_text)
        await send_message_to_lark(chat_id, chatgpt_reply)
    except Exception as e:
        logger.error(f"與外部 API 互動時出錯: {e}", exc_info=True)
        # 通知使用者發生錯誤
        await send_message_to_lark(chat_id, "抱歉，我現在遇到一點問題，請稍後再試。")


async def get_chatgpt_response(prompt: str) -> str:
    """
    向 OpenAI API 發送請求並獲取回應。
    """
    logger.info(f"向 OpenAI 發送請求: '{prompt[:50]}...'")
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
        response.raise_for_status() # 確保請求成功
        result = response.json()
        reply_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not reply_text:
            logger.warning("OpenAI 返回了空的回應。")
            return "抱歉，我思考了一下，但沒有找到答案。"
        return reply_text


async def get_lark_token() -> str:
    """
    獲取 Lark 應用所需的 tenant_access_token。
    """
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
    """
    向指定的 Lark 聊天發送文字訊息。
    """
    logger.info(f"準備向 chat_id {chat_id} 發送訊息: '{text[:50]}...'")
    try:
        token = await get_lark_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        # --- 【關鍵修正】content 欄位必須是一個 JSON 字串 ---
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
                logger.error(f"發送 Lark 訊息失敗: Code={result.get('code')}, Msg={result.get('msg')}")
    except Exception as e:
        logger.error(f"發送訊息到 Lark 時發生嚴重錯誤: {e}", exc_info=True)

