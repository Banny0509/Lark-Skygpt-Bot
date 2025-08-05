import os
import json
import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")
TENANT_ACCESS_TOKEN = os.getenv("TENANT_ACCESS_TOKEN")
VERIFICATION_TOKEN = os.getenv("VERIFICATION_TOKEN")


@app.post("/webhook")
async def webhook(request: Request):
    try:
        body = await request.json()
        print("Received body:", json.dumps(body))

        # Lark 驗證用
        if body.get("type") == "url_verification":
            if body.get("token") != VERIFICATION_TOKEN:
                return JSONResponse(status_code=403, content={"error": "Invalid verification token"})
            return {"challenge": body.get("challenge")}

        # 訊息事件處理
        if body.get("header", {}).get("event_type") == "im.message.receive_v1":
            message_content = json.loads(body["event"]["message"]["content"])
            user_id = body["event"]["sender"]["sender_id"]["user_id"]
            msg_type = body["event"]["message"]["message_type"]
            chat_id = body["event"]["message"]["chat_id"]

            if msg_type == "text":
                user_message = message_content.get("text", "")

                # 與 ChatGPT 溝通
                try:
                    gpt_reply = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "你是貼心的工作助理，請根據使用者指示回覆具體資訊。"},
                            {"role": "user", "content": user_message}
                        ]
                    )
                    reply_text = gpt_reply.choices[0].message.content
                except Exception as e:
                    reply_text = "很抱歉，目前無法提供服務，請稍後再試。"
                    print("OpenAI 回覆錯誤:", e)

                # 傳送回 Lark
                try:
                    headers = {
                        "Authorization": f"Bearer {TENANT_ACCESS_TOKEN}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "chat_id": chat_id,
                        "msg_type": "text",
                        "content": json.dumps({"text": reply_text})
                    }
                    async with httpx.AsyncClient() as client:
                        res = await client.post(
                            "https://open.larksuite.com/open-apis/im/v1/messages",
                            headers=headers,
                            json=payload
                        )
                        print("Lark 發送結果:", res.status_code, await res.aread())
                except Exception as e:
                    print("傳送訊息給 Lark 發生錯誤:", e)

        return {"code": 0, "msg": "OK"}
    except Exception as e:
        print("Webhook 處理錯誤:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
