from fastapi import FastAPI, Request
import httpx
import os
from dotenv import load_dotenv
import json
import logging

# --- 日誌紀錄設定 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# --- 從環境變數讀取設定 ---
LARK_APP_ID = os.getenv("APP_ID")
LARK_APP_SECRET = os.getenv("APP_SECRET")
VERIFICATION_TOKEN = os.getenv("VERIFICATION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.post("/webhook")
async def webhook(request: Request):
    """
    處理來自 Lark (飛書) 平台的 Webhook 請求。
    Handles incoming webhooks from the Lark platform.
    """
    payload = await request.json()

    # --- Webhook 驗證挑戰 ---
    if "challenge" in payload:
        return {"challenge": payload["challenge"]}

    # --- Token 驗證 ---
    header = payload.get("header", {})
    if header.get("token") != VERIFICATION_TOKEN:
        logger.warning("Invalid token received.")
        return {"code": 1, "message": "Invalid token"}

    # --- 事件處理 ---
    event_type = header.get("event_type")
    if event_type == "im.message.receive_v1":
        try:
            message = payload.get("event", {}).get("message", {})
            msg_type = message.get("message_type")

            # 只處理文字 (text) 類型的訊息
            if msg_type != "text":
                logger.info(f"Ignoring non-text message type: {msg_type}")
                return {"code": 0, "message": "ok"}

            chat_id = message.get("chat_id")
            content_str = message.get("content", "{}")

            # 將 JSON 解析包在 try-except 中
            content_dict = json.loads(content_str)
            text_raw = content_dict.get("text", "")
            
            # 清理 @all 標籤並去除頭尾空格
            user_text = text_raw.replace('<at user_id=\"all\">所有人</at>', '').strip()

            # 如果處理後訊息為空，則不呼叫 OpenAI
            if not user_text:
                logger.info("Empty message after processing, ignoring.")
                return {"code": 0, "message": "ok"}
            
            if not chat_id:
                logger.error("chat_id is missing in the payload.")
                return {"code": 1, "message": "Missing chat_id"}

            logger.info(f"Received from chat_id {chat_id}: {user_text}")
            reply = await get_chatgpt_response(user_text)
            await send_message_to_lark(chat_id, reply)

        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}. Content was: {content_str}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            logger.error(f"Problematic Payload: {payload}")

    return {"code": 0, "message": "ok"}

async def get_chatgpt_response(prompt: str) -> str:
    """
    從 OpenAI ChatGPT API 獲取回應。
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "抱歉，我遇到了一個錯誤。")
        except httpx.RequestError as e:
            logger.error(f"HTTP request to OpenAI failed: {e}")
            return "抱歉，我無法連接到 AI 服務。"
        except Exception as e:
            logger.error(f"An error occurred while getting ChatGPT response: {e}")
            return "抱歉，發生了未預期的錯誤。"

async def get_lark_token() -> str:
    """
    從 Lark 獲取 tenant_access_token。
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal",
            json={
                "app_id": LARK_APP_ID,
                "app_secret": LARK_APP_SECRET
            }
        )
        response.raise_for_status()
        return response.json()["tenant_access_token"]

async def send_message_to_lark(chat_id: str, text: str):
    """
    向指定的 Lark 聊天發送文字訊息。
    """
    try:
        token = await get_lark_token()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://open.larksuite.com/open-apis/im/v1/messages?receive_id_type=chat_id",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                json={
                    "receive_id": chat_id,
                    "msg_type": "text",
                    "content": json.dumps({"text": text})
                }
            )
            response.raise_for_status()
            logger.info(f"Sent message to chat_id {chat_id}. Response: {response.json()}")
    except Exception as e:
        logger.error(f"Failed to send message to Lark: {e}")

