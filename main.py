import httpx
import os
from dotenv import load_dotenv
import json
import logging
from fastapi import FastAPI, Request, HTTPException

# --- 1. 日誌設定 (非常重要) ---
# 設定日誌記錄，方便在 Railway 或其他平台上查看應用程式的運行狀況和錯誤。
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 2. 載入環境變數 ---
# 從 .env 檔案或平台設定的環境變數中讀取敏感資訊。
load_dotenv()

# --- 3. 初始化 FastAPI 應用 ---
app = FastAPI()

# --- 4. 讀取設定值 ---
LARK_APP_ID = os.getenv("APP_ID")
LARK_APP_SECRET = os.getenv("APP_SECRET")
VERIFICATION_TOKEN = os.getenv("VERIFICATION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 檢查必要的環境變數是否都已設定
if not all([LARK_APP_ID, LARK_APP_SECRET, VERIFICATION_TOKEN, OPENAI_API_KEY]):
    logger.critical("一個或多個必要的環境變數未設定！請檢查您的 .env 檔案或平台設定。")
    # 在這種嚴重情況下，可以選擇讓應用程式無法啟動
    # raise ValueError("一個或多個必要的環境變數未設定！")


@app.post("/webhook")
async def webhook(request: Request):
    """
    接收並處理來自 Lark (飛書) 開放平台的 Webhook 事件。
    """
    # --- 安全性：驗證請求來源 ---
    # 實際部署時，可以考慮增加對請求來源 IP 的驗證
    
    payload_bytes = await request.body()
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError:
        logger.error("收到了無效的 JSON 格式請求。")
        raise HTTPException(status_code=400, detail="無效的 JSON 格式。")

    # --- Lark Webhook URL 驗證挑戰 ---
    # 當您在 Lark 開發者後台設定 Webhook URL 時，Lark 會發送此請求。
    if "challenge" in payload:
        logger.info("收到 URL 驗證挑戰，已成功回應。")
        return {"challenge": payload["challenge"]}

    # --- 事件 Token 驗證 ---
    # 確保收到的事件是來自您自己的 Lark 應用，而不是惡意來源。
    header = payload.get("header", {})
    if header.get("token") != VERIFICATION_TOKEN:
        logger.warning(f"收到了無效的 Token: {header.get('token')}")
        raise HTTPException(status_code=403, detail="無效的 Token。")

    # --- 事件處理路由 ---
    event_type = header.get("event_type")
    if event_type == "im.message.receive_v1":
        try:
            await handle_message_receive(payload.get("event", {}))
        except Exception as e:
            # 捕獲處理過程中的任何意外錯誤
            logger.error(f"處理訊息時發生未預期錯誤: {e}", exc_info=True)
    else:
        logger.info(f"收到了未處理的事件類型: {event_type}")

    # 向 Lark 回應成功，避免重試攻擊
    return {"code": 0}


async def handle_message_receive(event: dict):
    """
    專門處理「接收訊息」事件的函式。
    """
    message = event.get("message", {})
    if not message:
        logger.warning("收到了空的 message 事件。")
        return

    message_type = message.get("message_type")
    chat_id = message.get("chat_id")
    
    # --- 【關鍵修正】只處理文字訊息 ---
    if message_type != "text":
        logger.info(f"忽略非文字訊息 (類型: {message_type})。")
        return

    # --- 【關鍵修正】正確解析 content 內容 ---
    # Lark 的 content 是一個 JSON 字串，必須先解析
    try:
        content_str = message.get("content", "{}")
        content_dict = json.loads(content_str)
        text_from_lark = content_dict.get("text", "")
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"解析訊息 content 失敗: {e}, 原始 content: {message.get('content')}")
        return

    # 清理 @機器人 和頭尾空格
    user_text = text_from_lark.replace(f'@_user_1', '').strip()

    # 如果訊息為空，則不處理
    if not user_text:
        logger.info("處理後訊息為空，已忽略。")
        return

    logger.info(f"從 chat_id {chat_id} 收到有效訊息: '{user_text}'")

    # 獲取 ChatGPT 回應並發送回 Lark
    try:
        chatgpt_reply = await get_chatgpt_response(user_text)
        await send_message_to_lark(chat_id, chatgpt_reply)
    except Exception as e:
        # 如果在與外部 API 互動時出錯，記錄錯誤並可選擇性地通知使用者
        logger.error(f"獲取 ChatGPT 回應或發送訊息時出錯: {e}", exc_info=True)
        # 可以考慮發送一條錯誤提示訊息給使用者
        await send_message_to_lark(chat_id, "抱歉，我現在遇到一點問題，請稍後再試。")


async def get_chatgpt_response(prompt: str) -> str:
    """
    向 OpenAI API 發送請求並獲取回應。
    """
    logger.info(f"向 OpenAI 發送請求: '{prompt[:50]}...'")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",  # 您可以根據需要更換模型
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=120  # 增加超時時間以應對可能較長的回應
            )
            response.raise_for_status()  # 如果 API 回傳非 2xx 狀態碼，會拋出異常
            result = response.json()
            reply_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.info(f"從 OpenAI 收到回應: '{reply_text[:50]}...'")
            if not reply_text:
                logger.warning("OpenAI 返回了空的回應。")
                return "抱歉，我沒有得到任何回應。"
            return reply_text
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI API 請求失敗，狀態碼: {e.response.status_code}, 回應: {e.response.text}")
            raise Exception("OpenAI API 請求失敗。")
        except httpx.RequestError as e:
            logger.error(f"無法連接到 OpenAI API: {e}")
            raise Exception("無法連接到 OpenAI API。")


async def get_lark_token() -> str:
    """
    獲取 Lark 企業自建應用的 tenant_access_token。
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": LARK_APP_ID, "app_secret": LARK_APP_SECRET}
        )
        response.raise_for_status()
        data = response.json()
        if "tenant_access_token" not in data:
            logger.error(f"獲取 Lark Token 失敗，回應: {data}")
            raise Exception("獲取 Lark Token 失敗。")
        return data["tenant_access_token"]


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
        payload = {
            "receive_id": chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}) # content 必須是 JSON 字串
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://open.larksuite.com/open-apis/im/v1/messages?receive_id_type=chat_id",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            if result.get("code") != 0:
                logger.error(f"發送 Lark 訊息失敗: Code={result.get('code')}, Msg={result.get('msg')}")
            else:
                logger.info("訊息已成功發送到 Lark。")
    except Exception as e:
        logger.error(f"發送訊息到 Lark 時發生嚴重錯誤: {e}", exc_info=True)
        # 此處的異常需要被上層捕獲

