from fastapi import FastAPI, Request
import httpx
import os
from dotenv import load_dotenv
import json
import logging

# --- 建議增加日誌紀錄，方便未來除錯 ---
# --- It is recommended to add logging for future debugging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

LARK_APP_ID = os.getenv("APP_ID")
LARK_APP_SECRET = os.getenv("APP_SECRET")
VERIFICATION_TOKEN = os.getenv("VERIFICATION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.post("/webhook")
async def webhook(request: Request):
    """
    Handles incoming webhooks from the Lark platform.
    """
    payload = await request.json()

    # --- Webhook 驗證 ---
    # --- Webhook verification challenge ---
    if "challenge" in payload:
        return {"challenge": payload["challenge"]}

    # --- Token 驗證 ---
    # --- Token validation ---
    header = payload.get("header", {})
    if header.get("token") != VERIFICATION_TOKEN:
        logger.warning("Invalid token received.")
        return {"code": 1, "message": "Invalid token"}

    # --- 事件處理 ---
    # --- Event handling ---
    event_type = header.get("event_type")
    if event_type == "im.message.receive_v1":
        try:
            message = payload.get("event", {}).get("message", {})
            msg_type = message.get("message_type")

            # 1.【修正】只處理文字 (text) 類型的訊息
            # 1. [FIX] Only process messages of type "text"
            if msg_type != "text":
                logger.info(f"Ignoring non-text message type: {msg_type}")
                return {"code": 0, "message": "ok"}

            chat_id = message.get("chat_id")
            content_str = message.get("content", "{}")

            # 2.【修正】將 JSON 解析包在 try-except 中，並處理空內容
            # 2. [FIX] Wrap JSON parsing in a try-except block and handle empty content
            content_dict = json.loads(content_str)
            text_raw = content_dict.get("text", "")
            
            # 清理 @all 標籤並去除頭尾空格
            # Clean up the @all tag and strip whitespace
            user_text = text_raw.replace('<at user_id=\"all\">所有人</at>', '').strip()

            # 3.【修正】如果處理後訊息為空，則不呼叫 OpenAI
            # 3. [FIX] If the message is empty after processing, do not call OpenAI
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
            return {"code": 1, "message": "JSON Decode Error"}
        except Exception as e:
            # 捕獲其他所有潛在錯誤
            # Catch all other potential errors
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            logger.error(f"Problematic Payload: {payload}")
            return {"code": 1, "message": "Internal Server Error"}

    return {"code": 0, "message": "ok"}

async def get_chatgpt_response(prompt: str) -> str:
    """
    Gets a response from the OpenAI ChatGPT API.
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
                timeout=60 # 建議增加超時時間，避免 OpenAI 回應過久 (Recommended to add a timeout)
            )
            response.raise_for_status() # 如果 API 回傳非 2xx 狀態碼，會拋出異常 (Raises an exception for non-2xx status codes)
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "Sorry, I encountered an error.")
        except httpx.RequestError as e:
            logger.error(f"HTTP request to OpenAI failed: {e}")
            return "Sorry, I couldn't connect to the AI service."
        except Exception as e:
            logger.error(f"An error occurred while getting ChatGPT response: {e}")
            return "Sorry, an unexpected error occurred."


async def get_lark_token() -> str:
    """
    Retrieves the tenant_access_token from Lark.
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
    Sends a text message to a specified Lark chat.
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

