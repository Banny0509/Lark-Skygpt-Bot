import httpx
import os
import json
import logging
import re
from fastapi import FastAPI, Request, HTTPException
# 引入日期和時間工具
from datetime import datetime, timezone, timedelta

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
        return {"challenge": payload["challenge"]}

    header = payload.get("header", {})
    if header.get("token") != VERIFICATION_TOKEN:
        raise HTTPException(status_code=403, detail="無效的 Token。")

    event_type = header.get("event_type")
    if event_type == "im.message.receive_v1":
        await handle_message_receive(payload.get("event", {}))

    return {"code": 0}

# --- 定義我們的「工具箱」 ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time_and_date",
            "description": "當使用者詢問任何關於『今天』、『現在』的日期、時間或星期幾時，使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone_offset": {
                        "type": "integer",
                        "description": "時區偏移量，以小時為單位。例如，北京時間 (UTC+8) 請輸入 8。",
                        "default": 8
                    },
                },
            },
        }
    }
]

# --- 我們的工具函式 ---
def get_current_time_and_date(timezone_offset: int = 8) -> str:
    """獲取指定時區的當前日期和時間。"""
    try:
        tz = timezone(timedelta(hours=timezone_offset))
        now = datetime.now(tz)
        return now.strftime(f"好的，目前時間是 %Y年%m月%d日 星期%A %H:%M:%S (UTC+{timezone_offset})。")
    except Exception as e:
        logger.error(f"獲取時間時發生錯誤: {e}")
        return "抱歉，我無法獲取當前的時間。"


async def handle_message_receive(event: dict):
    """
    專門處理「接收訊息」事件，並整合了「場景判斷」、「條件觸發」和「工具使用」的邏輯。
    """
    message = event.get("message", {})
    if not message:
        return

    chat_type = message.get("chat_type")
    
    # --- 【關鍵修正】判斷聊天類型 ---
    if chat_type == "group":
        # 如果是群聊，必須要 @機器人
        mentions = message.get("mentions")
        if not mentions:
            logger.info("群聊中沒有提及任何人，已忽略。")
            return

        is_bot_mentioned = False
        for mention in mentions:
            mentioned_id = mention.get("id", {}).get("user_id")
            if mentioned_id == LARK_APP_ID:
                is_bot_mentioned = True
                break

        if not is_bot_mentioned:
            logger.info("群聊中機器人未被提及，已忽略。")
            return
        
        logger.info("偵測到機器人在群聊中被提及，開始處理訊息...")

    elif chat_type == "p2p":
        # 如果是私聊，直接處理
        logger.info("偵測到私聊訊息，開始處理...")
    
    else:
        # 其他聊天類型暫不處理
        logger.info(f"收到未處理的聊天類型: {chat_type}，已忽略。")
        return

    # --- 後續處理邏輯 ---
    chat_id = message.get("chat_id")
    message_type = message.get("message_type")
    
    if message_type != "text":
        logger.info(f"訊息非文字類型，已忽略。")
        return

    try:
        content_str = message.get("content", "{}")
        content_dict = json.loads(content_str)
        text_from_lark = content_dict.get("text", "")
    except Exception as e:
        logger.error(f"解析訊息 content 失敗: {e}")
        return

    user_text = re.sub(r'<at.*?</at>', '', text_from_lark).strip()
    if not user_text:
        logger.info("移除 @提及 後訊息為空，已忽略。")
        return

    logger.info(f"收到來自 {chat_id} 的有效問題: '{user_text}'")

    try:
        messages = [
            {"role": "system", "content": "你是一個名叫『Lark-Skygpt-Bot』的專業 AI 助手。你的知識截止於 2023 年，所以任何關於即時資訊（例如今天日期、現在時間）的問題，你都必須使用 `get_current_time_and_date` 工具來查詢。在回答問題時，請務必簡潔、專業。"},
            {"role": "user", "content": user_text}
        ]

        response_json = await call_openai_api(messages, tools)
        response_message = response_json["choices"][0]["message"]

        if response_message.get("tool_calls"):
            logger.info("AI 決定使用工具...")
            messages.append(response_message)
            
            tool_call = response_message["tool_calls"][0]
            function_name = tool_call["function"]["name"]
            
            if function_name == "get_current_time_and_date":
                function_response = get_current_time_and_date()
                messages.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })

                logger.info("將工具結果送回 AI 進行總結...")
                final_response_json = await call_openai_api(messages)
                final_answer = final_response_json["choices"][0]["message"]["content"]
            else:
                final_answer = "抱歉，我不知道如何使用這個工具。"
        else:
            logger.info("AI 決定不使用工具，直接回答。")
            final_answer = response_message["content"]

        await send_message_to_lark(chat_id, final_answer)

    except Exception as e:
        logger.error(f"處理訊息時發生最上層錯誤: {e}", exc_info=True)
        await send_message_to_lark(chat_id, "抱歉，處理您的請求時發生了未預期的錯誤。")


async def call_openai_api(messages, tools=None):
    """一個專門用來呼叫 OpenAI API 的函式。"""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": "gpt-4-turbo",
        "messages": messages
    }
    if tools:
        json_data["tools"] = tools
        json_data["tool_choice"] = "auto"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
            timeout=120
        )
        response.raise_for_status()
        return response.json()


async def get_lark_token() -> str:
    # ... 此函式內容與之前相同 ...
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
    # ... 此函式內容與之前相同 ...
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
                logger.error(f"發送 Lark 訊息失敗: {result}")
    except Exception as e:
        logger.error(f"發送訊息到 Lark 時發生嚴重錯誤: {e}", exc_info=True)
