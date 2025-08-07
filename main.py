import httpx
import os
import json
import logging
import re
from fastapi import FastAPI, Request, HTTPException

# --- 【新功能】引入 Lark Base API 的函式庫 ---
# 您需要先在 requirements.txt 中加入 lark-oapi
# 然後執行 pip install -r requirements.txt
import lark_oapi as lark
from lark_oapi.api.base.v1 import ListAppTableRecordRequest, ListAppTableRecordResponse

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
# --- 【新功能】為您的多維表格設定環境變數 ---
LARK_BASE_APP_TOKEN = os.getenv("LARK_BASE_APP_TOKEN") # 您多維表格的 AppToken
LARK_BASE_TABLE_ID = os.getenv("LARK_BASE_TABLE_ID")   # 您要查詢的表格的 TableId

if not all([LARK_APP_ID, LARK_APP_SECRET, VERIFICATION_TOKEN, OPENAI_API_KEY, LARK_BASE_APP_TOKEN, LARK_BASE_TABLE_ID]):
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

# --- 【新功能】定義我們的「工具箱」 ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_customer_data_from_lark_base",
            "description": "當使用者想要查詢客戶資料時，使用此工具。例如查詢電話、地址、訂單狀態等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {
                        "type": "string",
                        "description": "要查詢的客戶姓名，例如 'Banny' 或 '陳先生'",
                    },
                },
                "required": ["customer_name"],
            },
        }
    }
]

# --- 【新功能】我們的工具函式 ---
async def query_customer_data_from_lark_base(customer_name: str) -> str:
    """
    這是一個範例工具，它會去 Lark 多維表格中查詢客戶資料。
    注意：這段程式碼是一個簡化的範例，您需要根據您表格的實際欄位進行修改。
    """
    logger.info(f"--- 正在執行工具：在 Lark Base 中查詢客戶 '{customer_name}' ---")
    try:
        # 1. 建立 Lark API 客戶端
        client = lark.Client.builder().app_id(LARK_APP_ID).app_secret(LARK_APP_SECRET).build()

        # 2. 建立請求
        # 我們使用 'filter' 參數來篩選出我們想要的客戶
        # 這裡的 CurrentValue.[客戶姓名欄位] 需要換成您表格中客戶姓名的真實欄位名稱
        request = ListAppTableRecordRequest.builder() \
            .app_token(LARK_BASE_APP_TOKEN) \
            .table_id(LARK_BASE_TABLE_ID) \
            .filter(f"CurrentValue.[客戶姓名]=\"{customer_name}\"") \
            .page_size(1) \
            .build()

        # 3. 發送 API 請求
        response: ListAppTableRecordResponse = client.base.v1.app_table_record.list(request)

        # 4. 處理回應
        if not response.success():
            logger.error(f"Lark Base API 錯誤: {response.code}, {response.msg}")
            return f"查詢失敗：無法連接到資料庫。"
        
        if response.data and response.data.items:
            record = response.data.items[0]
            # 將查詢到的欄位組合成一個字串回傳
            # 您需要將 'fields' 中的 key 換成您表格的真實欄位名稱
            fields = record.fields
            result_str = f"客戶姓名: {fields.get('客戶姓名')}, 電話: {fields.get('電話')}, 最新訂單狀態: {fields.get('訂單狀態')}"
            logger.info(f"查詢成功，結果: {result_str}")
            return result_str
        else:
            logger.info("在資料庫中未找到該客戶。")
            return f"查詢失敗：在資料庫中找不到名為 '{customer_name}' 的客戶。"

    except Exception as e:
        logger.error(f"執行 query_customer_data_from_lark_base 工具時發生錯誤: {e}", exc_info=True)
        return "查詢時發生內部錯誤。"


async def handle_message_receive(event: dict):
    message = event.get("message", {})
    if not (message and message.get("message_type") == "text"):
        return

    chat_id = message.get("chat_id")
    try:
        content_dict = json.loads(message.get("content", "{}"))
        text_from_lark = content_dict.get("text", "")
    except Exception:
        return

    user_text = re.sub(r'<at.*?</at>', '', text_from_lark).strip()
    if not user_text:
        return

    logger.info(f"收到來自 {chat_id} 的問題: '{user_text}'")

    try:
        # --- 【新架構】與 AI 的多輪對話循環 ---
        
        # 1. 準備好我們的對話歷史
        messages = [
            {"role": "system", "content": "你是一個名叫『Lark-Skygpt-Bot』的專業、簡潔、高效的 AI 助手。你的任務是精確地回答使用者的問題，並在必要時使用工具來獲取資訊。不要說任何與任務無關的閒聊或客套話。"},
            {"role": "user", "content": user_text}
        ]

        # 2. 第一次呼叫 AI，讓它思考
        response = await call_openai_api(messages, tools)
        response_message = response.choices[0].message

        # 3. 檢查 AI 是否需要使用工具
        tool_calls = response_message.tool_calls
        if tool_calls:
            logger.info("AI 決定使用工具...")
            # 將 AI 的回應加入對話歷史
            messages.append(response_message)

            # 4. 執行所有 AI 要求使用的工具
            available_functions = {
                "query_customer_data_from_lark_base": query_customer_data_from_lark_base,
            }
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                # 執行工具函式
                function_response = await function_to_call(
                    customer_name=function_args.get("customer_name")
                )
                
                # 將工具的執行結果加入對話歷史
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
            
            # 5. 再次呼叫 AI，讓它根據工具的結果進行總結
            logger.info("將工具結果送回 AI 進行總結...")
            final_response = await call_openai_api(messages)
            final_answer = final_response.choices[0].message.content

        else:
            # 如果 AI 不需要使用工具，直接使用它的第一次回應
            logger.info("AI 決定不使用工具，直接回答。")
            final_answer = response_message.content

        # 6. 將最終答案發送給使用者
        await send_message_to_lark(chat_id, final_answer)

    except Exception as e:
        logger.error(f"處理訊息時發生最上層錯誤: {e}", exc_info=True)
        await send_message_to_lark(chat_id, "抱歉，處理您的請求時發生了未預期的錯誤。")


async def call_openai_api(messages, tools=None):
    """
    一個專門用來呼叫 OpenAI API 的函式。
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": "gpt-4-turbo", # 推薦使用支援工具呼叫的最新模型
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
    """獲取 Lark 應用所需的 tenant_access_token。"""
    # ... 此函式內容與之前相同，為節省篇幅已省略 ...
    # 請確保此函式可以正常運作
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
    """向指定的 Lark 聊天發送文字訊息。"""
    # ... 此函式內容與之前相同，為節省篇幅已省略 ...
    # 請確保此函式可以正常運作
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
                logger.info(f"訊息已成功發送到 {chat_id}。")
            else:
                logger.error(f"發送 Lark 訊息失敗: {result}")
    except Exception as e:
        logger.error(f"發送訊息到 Lark 時發生嚴重錯誤: {e}", exc_info=True)

