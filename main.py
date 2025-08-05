from fastapi import FastAPI, Request
import httpx
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()
app = FastAPI()

LARK_APP_ID = os.getenv("APP_ID")
LARK_APP_SECRET = os.getenv("APP_SECRET")
VERIFICATION_TOKEN = os.getenv("VERIFICATION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()

    # ✅ 回應 Lark，避免 webhook timeout
    response = {"code": 0, "message": "ok"}

    try:
        if payload.get("header", {}).get("token") != VERIFICATION_TOKEN:
            return {"code": 1, "message": "Invalid token"}

        event_type = payload.get("header", {}).get("event_type")
        if event_type == "im.message.receive_v1":
            message = payload["event"]["message"]
            chat_id = message["chat_id"]
            text_raw = message["content"]
            user_text = text_raw.replace('<at user_id="all">所有人</at>', '').strip()

            # ✅ 非同步處理，不阻塞回應
            asyncio.create_task(process_message(chat_id, user_text))

    except Exception as e:
        print(f"Webhook error: {e}")

    return response


async def process_message(chat_id: str, user_text: str):
    reply = await get_chatgpt_response(user_text)
    await send_message_to_lark(chat_id, reply)


async def get_chatgpt_response(prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        result = response.json()
        return result["choices"][0]["message"]["content"]


async def send_message_to_lark(chat_id: str, text: str):
    token = await get_lark_token()
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://open.larksuite.com/open-apis/im/v1/messages?receive_id_type=chat_id",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json={
                "receive_id": chat_id,
                "msg_type": "text",
                "content": {"text": text}
            }
        )


async def get_lark_token() -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal",
            json={
                "app_id": LARK_APP_ID,
                "app_secret": LARK_APP_SECRET
            }
        )
        return response.json()["tenant_access_token"]

