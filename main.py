
from fastapi import FastAPI, Request
import httpx
import os
from dotenv import load_dotenv
import json

load_dotenv()

app = FastAPI()

LARK_APP_ID = os.getenv("APP_ID")
LARK_APP_SECRET = os.getenv("APP_SECRET")
VERIFICATION_TOKEN = os.getenv("VERIFICATION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()

    # webhook 驗證階段
    if "challenge" in payload:
        return {"challenge": payload["challenge"]}

    # token 驗證
    if payload.get("header", {}).get("token") != VERIFICATION_TOKEN:
        return {"code": 1, "message": "Invalid token"}

    # 處理收到訊息
    event_type = payload.get("header", {}).get("event_type")
    if event_type == "im.message.receive_v1":
        message = payload["event"]["message"]
        chat_id = message["chat_id"]
        text_raw = json.loads(message["content"]).get("text", "")
        user_text = text_raw.replace('<at user_id=\"all\">所有人</at>', '').strip()

        reply = await get_chatgpt_response(user_text)
        await send_message_to_lark(chat_id, reply)

    return {"code": 0, "message": "ok"}

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
                "content": json.dumps({"text": text})
            }
        )
