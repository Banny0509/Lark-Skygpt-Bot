# Copy of original main.py from repository with modifications to support multi‑modal file and image messages.
import httpx
import os
import json
import logging
import re
import base64
import mimetypes
from io import BytesIO
from datetime import datetime, timezone, timedelta
from typing import Optional

# Third‑party libraries used for parsing various document types.  These will be
# declared in requirements.txt so they are available in your environment.
try:
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None
try:
    import openpyxl  # type: ignore
except ImportError:
    openpyxl = None
try:
    import docx  # type: ignore
except ImportError:
    docx = None  # python‑docx
try:
    import pptx  # type: ignore
except ImportError:
    pptx = None  # python‑pptx
try:
    from PIL import Image  # type: ignore
except ImportError:
    Image = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Determine whether to load .env locally or rely on Railway's environment.
if "RAILWAY_ENVIRONMENT" not in os.environ:
    from dotenv import load_dotenv  # type: ignore
    logger.info("偵測到非生產環境，正在載入 .env 檔案...")
    load_dotenv()
else:
    logger.info("偵測到生產環境，將直接使用平台設定的環境變數。")

from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

# Load required environment variables
LARK_APP_ID = os.getenv("APP_ID")
LARK_APP_SECRET = os.getenv("APP_SECRET")
VERIFICATION_TOKEN = os.getenv("VERIFICATION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([LARK_APP_ID, LARK_APP_SECRET, VERIFICATION_TOKEN, OPENAI_API_KEY]):
    logger.critical("一個或多個必要的環境變數未設定！")

@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok"}

@app.post("/webhook")
async def webhook(request: Request):
    """Primary entrypoint for Lark webhook events.  Validates the verification
    token, handles initial handshake and delegates message processing."""
    payload_bytes = await request.body()
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError:
        logger.error("收到了無效的 JSON 格式請求。")
        raise HTTPException(status_code=400, detail="無效的 JSON 格式。")

    # Handle initial handshake from Lark
    if "challenge" in payload:
        return {"challenge": payload["challenge"]}

    header = payload.get("header", {})
    if header.get("token") != VERIFICATION_TOKEN:
        raise HTTPException(status_code=403, detail="無效的 Token。")

    event_type = header.get("event_type")
    if event_type == "im.message.receive_v1":
        await handle_message_receive(payload.get("event", {}))

    return {"code": 0}

# ---------------------------------------------------------------------------
# Tools and tool functions
# ---------------------------------------------------------------------------
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

def get_current_time_and_date(timezone_offset: int = 8) -> str:
    """獲取指定時區的當前日期和時間。"""
    try:
        tz = timezone(timedelta(hours=timezone_offset))
        now = datetime.now(tz)
        return now.strftime(f"好的，目前時間是 %Y年%m月%d日 星期%A %H:%M:%S (UTC+{timezone_offset})。")
    except Exception as e:
        logger.error(f"獲取時間時發生錯誤: {e}")
        return "抱歉，我無法獲取當前的時間。"

# ---------------------------------------------------------------------------
# Helper functions for Lark API interactions
# ---------------------------------------------------------------------------
async def get_lark_token() -> str:
    """Fetch a tenant access token from Lark.  Cached tokens may be reused in
    production, but this example fetches a fresh token each time."""
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
    """Send a plain text reply back to the originating chat."""
    try:
        token = await get_lark_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        payload = {
            "receive_id": chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False)
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

async def download_message_resource(message_id: str, file_key: str) -> bytes:
    """
    Download a binary resource (image/file) from a Lark message.  This wraps the
    official endpoint documented as:
      GET /open-apis/im/v1/messages/:message_id/resources/:file_key

    For full functionality your bot must have the appropriate permissions
    enabled in the developer console.
    """
    token = await get_lark_token()
    headers = {
        "Authorization": f"Bearer {token}"
    }
    url = f"https://open.larksuite.com/open-apis/im/v1/messages/{message_id}/resources/{file_key}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.content

def extract_text_from_file(file_bytes: bytes, file_name: str) -> str:
    """
    Convert various office document types (Excel, Word, PowerPoint, PDF) into
    plain text.  Falls back to a placeholder description if the format cannot
    be parsed or if the necessary parser library is missing.
    """
    _, ext = os.path.splitext(file_name)
    ext = ext.lower()
    try:
        if ext in {".xlsx", ".xls"} and openpyxl is not None:
            wb = openpyxl.load_workbook(filename=BytesIO(file_bytes), data_only=True, read_only=True)
            text_lines = []
            # Extract the first worksheet
            sheet = wb[wb.sheetnames[0]]
            for row in sheet.iter_rows(values_only=True):
                # Convert each row to tab-separated string and append
                line = "\t".join([str(cell) if cell is not None else "" for cell in row])
                text_lines.append(line)
            return "\n".join(text_lines)
        elif ext == ".docx" and docx is not None:
            document = docx.Document(BytesIO(file_bytes))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        elif ext in {".pptx", ".ppt"} and pptx is not None:
            prs = pptx.Presentation(BytesIO(file_bytes))
            texts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        texts.append(shape.text)
            return "\n".join(texts)
        elif ext == ".pdf" and pdfplumber is not None:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                pages_text = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
            return "\n".join(pages_text)
    except Exception as e:
        logger.error(f"解析 {file_name} 時發生錯誤: {e}", exc_info=True)
        return f"[無法解析 {file_name}，發生錯誤: {e}]"
    # Fallback for unsupported file types
    return f"[不支援的檔案類型 {ext}，僅支援 Excel、Word、PowerPoint、PDF。]"

def build_image_message(base64_str: str) -> list:
    """
    Construct a multi‑modal message payload for the OpenAI Chat API when sending
    images.  The `detail` field can be set to 'low' or 'high' depending on your
    quota and required resolution.  Here we default to 'low' to minimize cost.
    """
    return [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_str}",
                "detail": "low"
            }
        },
        {
            "type": "text",
            "text": "請閱讀這張圖片並根據其內容回答我的問題或提供摘要。"
        }
    ]

async def call_openai_api(
    messages,
    tools: Optional[list] = None,
    model: Optional[str] = None
) -> dict:
    """
    Generic wrapper around the OpenAI Chat Completion API.  Supports optional
    tools for function calling and allows overriding the model on a per‑call
    basis.  For multi‑modal interactions specify a model with vision
    capabilities such as 'gpt-4o'.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": model or "gpt-4-turbo",
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

# ---------------------------------------------------------------------------
# Message handling
# ---------------------------------------------------------------------------
async def handle_message_receive(event: dict):
    """
    Handle the 'im.message.receive_v1' event.  Supports private chats and group
    mentions.  Now includes rudimentary handling for images and files in
    addition to plain text messages.
    """
    message = event.get("message", {})
    if not message:
        return

    chat_type = message.get("chat_type")
    chat_id = message.get("chat_id")

    # Determine whether the message should trigger the bot
    is_trigger_condition_met = False
    if chat_type == "p2p":
        is_trigger_condition_met = True  # Private chats always trigger
        logger.info("偵測到私人聊天，將直接處理。")
    elif chat_type == "group":
        mentions = message.get("mentions")
        if mentions:
            for mention in mentions:
                mentioned_id = mention.get("id", {}).get("user_id")
                if mentioned_id == LARK_APP_ID:
                    is_trigger_condition_met = True
                    break

    if not is_trigger_condition_met:
        logger.info(f"位於群組 ({chat_id}) 的訊息未提及機器人，或是不支援的聊天類型，已忽略。")
        return

    # Once triggered, branch by message type
    message_type = message.get("message_type")
    logger.info(f"處理訊息類型: {message_type}")

    # Handle image messages
    if message_type == "image":
        await handle_image_message(message, chat_id)
        return

    # Handle file messages (Excel, Word, PowerPoint, PDF, etc.)
    if message_type == "file":
        await handle_file_message(message, chat_id)
        return

    # Only process text messages if type is exactly 'text'
    if message_type != "text":
        logger.info("訊息非文字類型，已忽略。")
        return

    # Extract user text from the Lark payload
    try:
        content_str = message.get("content", "{}")
        content_dict = json.loads(content_str)
        text_from_lark = content_dict.get("text", "")
    except Exception as e:
        logger.error(f"解析訊息 content 失敗: {e}")
        return

    # Remove mention tags in group chats
    user_text = re.sub(r'\<at[^>]*>.*?</at>', '', text_from_lark).strip()
    if not user_text:
        logger.info("移除 @提及 後訊息為空，已忽略。")
        return

    logger.info(f"收到來自 {chat_id} 的有效問題: '{user_text}'")

    try:
        # Compose initial messages for ChatGPT
        messages_payload = [
            {
                "role": "system",
                "content": (
                    "你是一個名叫『Lark-Skygpt-Bot』的專業 AI 助手。你的知識截止於 2023 年，所以任何"
                    "關於即時資訊（例如今天日期、現在時間）的問題，你都必須使用 `get_current_time_and_date` "
                    "工具來查訊。在回答問題時，請加量要簡潔、專業。"
                )
            },
            {"role": "user", "content": user_text}
        ]

        response_json = await call_openai_api(messages_payload, tools)
        response_message = response_json["choices"][0]["message"]

        if response_message.get("tool_calls"):
            logger.info("AI 決定使用工具...")
            messages_payload.append(response_message)
            tool_call = response_message["tool_calls"][0]
            function_name = tool_call["function"]["name"]

            if function_name == "get_current_time_and_date":
                function_response = get_current_time_and_date()
                messages_payload.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })
                logger.info("將工具結果送回 AI 進行總結...")
                final_response_json = await call_openai_api(messages_payload)
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

async def handle_image_message(message: dict, chat_id: str):
    """
    Download the image attached to the message, send it to a multimodal OpenAI
    model and forward the response back to the user.
    """
    file_key = message.get("file_key")
    message_id = message.get("message_id")
    if not (file_key and message_id):
        logger.error("缺少 file_key 或 message_id，無法下載圖片。")
        return

    try:
        file_bytes = await download_message_resource(message_id, file_key)
        # Encode the image as base64 for the vision API
        encoded = base64.b64encode(file_bytes).decode("utf-8")
        user_payload = build_image_message(encoded)
        messages_payload = [
            {
                "role": "system",
                "content": (
                    "你是一個具有強大圖像理解能力的 AI 助手，能夠分析圖片中包含的文字、表格、圖表"
                    "以及其他元素。請根據圖片內容完整回答用戶問題或提供摘要。"
                )
            },
            {
                "role": "user",
                "content": user_payload
            }
        ]
        response_json = await call_openai_api(messages_payload, model="gpt-4o")
        answer = response_json["choices"][0]["message"]["content"]
        await send_message_to_lark(chat_id, answer)
    except Exception as e:
        logger.error(f"處理圖片訊息時發生錯誤: {e}", exc_info=True)
        await send_message_to_lark(chat_id, "抱歉，處理圖片時發生錯誤。")

async def handle_file_message(message: dict, chat_id: str):
    """
    Download the attached file (Excel, Word, PowerPoint, PDF) from the message,
    extract its textual content and send it to ChatGPT for analysis.  The
    response from ChatGPT is then returned back to the Lark conversation.
    """
    file_key = message.get("file_key")
    file_name = message.get("file_name") or "unknown"
    message_id = message.get("message_id")
    if not (file_key and message_id):
        logger.error("缺少 file_key 或 message_id，無法下載檔案。")
        return
    try:
        file_bytes = await download_message_resource(message_id, file_key)
        extracted_text = extract_text_from_file(file_bytes, file_name)
        # Compose a prompt instructing the model to reason about the file content.
        messages_payload = [
            {
                "role": "system",
                "content": (
                    "你是一個能夠理解並分析檔案內容（包含 Excel、Word、PowerPoint、PDF）的 AI 助手。"
                    "請根據使用者提供的檔案內容進行摘要、回答問題或給出建議。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"以下是來自檔案《{file_name}》的提取內容：\n\n{extracted_text}\n\n"
                    "請閱讀上述內容並提供摘要或回答使用者的問題。"
                )
            }
        ]
        response_json = await call_openai_api(messages_payload, model="gpt-4o")
        answer = response_json["choices"][0]["message"]["content"]
        await send_message_to_lark(chat_id, answer)
    except Exception as e:
        logger.error(f"處理檔案訊息時發生錯誤: {e}", exc_info=True)
        await send_message_to_lark(chat_id, "抱歉，處理檔案時發生錯誤。")
