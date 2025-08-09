# Lark SkyGPT - FastAPI webhook with multi-modal, @mention gating in groups, P2P auto-reply,
# and daily 08:00 summary for previous day's group chats (Asia/Taipei by default).
import os
import json
import logging
import re
import base64
from io import BytesIO
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from collections import defaultdict
import asyncio
import sqlite3 # 用於持久化儲存

import httpx
from fastapi import FastAPI, Request, HTTPException

# Optional parsers for office/PDF and images
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None
try:
    import openpyxl  # type: ignore
except Exception:
    openpyxl = None
try:
    import docx  # type: ignore  # python-docx
except Exception:
    docx = None
try:
    import pptx  # type: ignore  # python-pptx
except Exception:
    pptx = None
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("lark-skygpt")

# Load .env locally when not on Railway (or other prod)
if "RAILWAY_ENVIRONMENT" not in os.environ:
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        logger.info("Loaded .env for local development")
    except Exception:
        pass

app = FastAPI()

# --- Environment ---
LARK_APP_ID = os.getenv("APP_ID")              # Bot APP_ID
LARK_APP_SECRET = os.getenv("APP_SECRET")      # Bot APP_SECRET
VERIFICATION_TOKEN = os.getenv("VERIFICATION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TIMEZONE_OFFSET = int(os.getenv("TIMEZONE_OFFSET", "8"))  # Asia/Taipei UTC+8 by default
DATABASE_FILE = "lark_chat_history.db" # 定義資料庫檔案名稱

if not all([LARK_APP_ID, LARK_APP_SECRET, VERIFICATION_TOKEN, OPENAI_API_KEY]):
    logger.warning("One or more required env vars are missing.")


# --- 資料庫相關函式 ---
def init_db():
    """初始化資料庫和資料表"""
    con = sqlite3.connect(DATABASE_FILE)
    cur = con.cursor()
    # 建立一個 message_id 為主鍵的資料表，並增加 summarized 欄位來追蹤是否已摘要
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            message_id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL,
            chat_type TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            text TEXT,
            summarized INTEGER DEFAULT 0
        )
    """)
    con.commit()
    con.close()
    logger.info("Database initialized successfully.")

def log_message_to_db(message_id: str, chat_id: str, chat_type: str, timestamp: datetime, text: str):
    """將訊息紀錄到資料庫"""
    con = sqlite3.connect(DATABASE_FILE)
    cur = con.cursor()
    try:
        cur.execute(
            "INSERT INTO chat_history (message_id, chat_id, chat_type, timestamp, text) VALUES (?, ?, ?, ?, ?)",
            (message_id, chat_id, chat_type, timestamp, text)
        )
        con.commit()
    except sqlite3.IntegrityError:
        logger.warning("Message with ID %s already exists.", message_id)
    except Exception as e:
        logger.exception("Failed to log message to DB: %s", e)
    finally:
        con.close()

def get_chats_for_summary(start_date: datetime, end_date: datetime) -> Dict[str, List[Dict]]:
    """從資料庫獲取需要摘要的群組聊天紀錄"""
    con = sqlite3.connect(DATABASE_FILE)
    con.row_factory = sqlite3.Row # 讓回傳結果可以像字典一樣用欄位名存取
    cur = con.cursor()
    cur.execute("""
        SELECT chat_id, timestamp, text FROM chat_history
        WHERE chat_type = 'group' AND summarized = 0 AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    """, (start_date, end_date))
    
    chats = defaultdict(list)
    rows = cur.fetchall()
    for row in rows:
        chats[row['chat_id']].append({"timestamp": datetime.fromisoformat(row['timestamp']), "text": row['text']})
    con.close()
    return chats

def mark_messages_as_summarized(start_date: datetime, end_date: datetime):
    """將已摘要的訊息在資料庫中標記"""
    con = sqlite3.connect(DATABASE_FILE)
    cur = con.cursor()
    cur.execute(
        "UPDATE chat_history SET summarized = 1 WHERE timestamp BETWEEN ? AND ?",
        (start_date, end_date)
    )
    con.commit()
    con.close()
    logger.info("Marked messages from %s to %s as summarized.", start_date, end_date)

# --- Health ---
@app.get("/health")
async def health():
    return {"status": "ok"}

# --- Lark helper functions ---
async def get_lark_token() -> str:
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": LARK_APP_ID, "app_secret": LARK_APP_SECRET},
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("tenant_access_token")
        if not token:
            raise RuntimeError(f"Failed to get tenant_access_token: {data}")
        return token

async def send_message_to_lark(chat_id: str, text: str) -> None:
    token = await get_lark_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    payload = {"receive_id": chat_id, "msg_type": "text", "content": json.dumps({"text": text}, ensure_ascii=False)}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://open.larksuite.com/open-apis/im/v1/messages?receive_id_type=chat_id",
            headers=headers, json=payload
        )
        try:
            r.raise_for_status()
        except Exception:
            logger.error("send_message_to_lark failed: %s", r.text)

async def download_message_resource(message_id: str, file_key: str) -> bytes:
    token = await get_lark_token()
    url = f"https://open.larksuite.com/open-apis/im/v1/messages/{message_id}/resources/{file_key}"
    headers = {"Authorization": f"Bearer {token}"}
    # 增加超時設定，避免無限期等待
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.content

# --- Parsing helpers ---
def _strip_mentions(text: str) -> str:
    # Remove Lark <at ...>...</at> tags
    return re.sub(r"<at[^>]*>.*?</at>", "", text or "").strip()

def extract_text_from_file(file_bytes: bytes, file_name: str) -> str:
    name_lower = (file_name or "").lower()
    try:
        if (name_lower.endswith(".xlsx") or name_lower.endswith(".xls")) and openpyxl is not None:
            wb = openpyxl.load_workbook(filename=BytesIO(file_bytes), data_only=True, read_only=True)
            sheet = wb[wb.sheetnames[0]]
            lines = []
            for row in sheet.iter_rows(values_only=True):
                line = "\t".join("" if c is None else str(c) for c in row)
                lines.append(line)
            return "\n".join(lines)
        if name_lower.endswith(".docx") and docx is not None:
            d = docx.Document(BytesIO(file_bytes))
            return "\n".join(p.text for p in d.paragraphs)
        if (name_lower.endswith(".pptx") or name_lower.endswith(".ppt")) and pptx is not None:
            prs = pptx.Presentation(BytesIO(file_bytes))
            out = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        out.append(shape.text)
            return "\n".join(out)
        if name_lower.endswith(".pdf") and pdfplumber is not None:
            texts = []
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        texts.append(t)
            return "\n".join(texts) if texts else "[空白 PDF 或無法抽取文字]"
    except Exception as e:
        logger.exception("extract_text_from_file error for %s: %s", file_name, e)
        return f"[解析 {file_name} 發生錯誤: {e}]"
    return f"[不支援的檔案或缺少解析庫: {file_name}]"

def build_image_content(b64_png: str, user_hint: str = "請閱讀此圖片並回覆重點或回答問題。") -> List[dict]:
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_png}", "detail": "low"}},
        {"type": "text", "text": user_hint},
    ]

# --- OpenAI ---
async def call_openai_api(messages: List[dict], model: Optional[str] = None) -> dict:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"model": model or OPENAI_MODEL, "messages": messages}
    # 增加超時設定，避免無限期等待
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        r.raise_for_status()
        return r.json()

# --- Webhook ---
@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Lark challenge handshake
    if "challenge" in payload:
        return {"challenge": payload["challenge"]}

    header = payload.get("header", {})
    if header.get("token") != VERIFICATION_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    if header.get("event_type") == "im.message.receive_v1":
        await handle_message_receive(payload.get("event", {}))

    return {"code": 0}

# --- Message handler with @-mention gating / P2P auto-reply / multi-modal ---
# ★★★ 以下是更新過的函式 ★★★
async def handle_message_receive(event: dict):
    message = event.get("message", {})
    if not message:
        return

    # --- 增加日誌，方便追蹤收到的原始訊息 ---
    logger.info("Received message payload: %s", json.dumps(message))

    chat_type = message.get("chat_type")        # "p2p" or "group"
    chat_id = message.get("chat_id")
    message_id = message.get("message_id")

    # timestamp (ms) -> local datetime
    create_time_ms = message.get("create_time")
    if create_time_ms:
        ts = datetime.fromtimestamp(int(create_time_ms) / 1000, tz=timezone.utc)
        ts_local = ts.astimezone(timezone(timedelta(hours=TIMEZONE_OFFSET)))
    else:
        ts_local = datetime.now(timezone(timedelta(hours=TIMEZONE_OFFSET)))

    msg_type = message.get("message_type")      # "text" | "image" | "file" ...
    content_raw = message.get("content") or "{}"
    try:
        content = json.loads(content_raw)
    except Exception:
        content = {}

    # 將訊息紀錄到資料庫而非記憶體
    summary_text = ""
    if msg_type == "text":
        summary_text = _strip_mentions(content.get("text", ""))
    elif msg_type == "image":
        summary_text = "[圖片]"
    elif msg_type == "file":
        summary_text = f"[檔案: {content.get('file_name') or ''}]"
    if all([message_id, chat_id, chat_type, summary_text]):
        log_message_to_db(message_id, chat_id, chat_type, ts_local, summary_text)

    # --- Trigger rules (使用更穩健的判斷邏輯) ---
    is_trigger = False
    if chat_type == "p2p":
        is_trigger = True
        logger.info("Triggered by P2P chat.")
    elif chat_type == "group":
        mentions = message.get("mentions", []) or []
        for m in mentions:
            # 防禦性檢查，涵蓋多種可能的 ID 格式
            id_union = m.get("id", {})
            possible_ids = set()
            if id_union.get("app_id"):
                possible_ids.add(id_union.get("app_id"))
            if id_union.get("open_id"):
                possible_ids.add(id_union.get("open_id"))

            if LARK_APP_ID in possible_ids:
                is_trigger = True
                logger.info("Triggered by @mention in group.")
                break
    
    if not is_trigger:
        logger.info("Ignored message (no @mention in group or not P2P).")
        return

    # --- Branch by message type ---
    try:
        if msg_type == "image":
            image_key = content.get("image_key") or content.get("file_key")
            if not image_key:
                await send_message_to_lark(chat_id, "收到圖片但缺少 image_key。")
                return
            file_bytes = await download_message_resource(message.get("message_id"), image_key)
            b64 = base64.b64encode(file_bytes).decode("utf-8")
            mm = [{"role": "user", "content": build_image_content(b64)}]
            resp = await call_openai_api(mm)
            reply = resp["choices"][0]["message"]["content"].strip()
            await send_message_to_lark(chat_id, reply)
            return

        if msg_type == "file":
            file_key = content.get("file_key")
            file_name = content.get("file_name") or "附件"
            if not file_key:
                await send_message_to_lark(chat_id, "收到檔案但缺少 file_key。")
                return
            file_bytes = await download_message_resource(message.get("message_id"), file_key)
            extracted = extract_text_from_file(file_bytes, file_name)
            prompt = f"請閱讀以下檔案內容，使用繁體中文摘要重點：\n\n{extracted[:15000]}"
            mm = [{"role": "user", "content": prompt}]
            resp = await call_openai_api(mm)
            reply = resp["choices"][0]["message"]["content"].strip()
            await send_message_to_lark(chat_id, reply)
            return

        if msg_type == "text":
            user_text = _strip_mentions(content.get("text", ""))
            if not user_text:
                await send_message_to_lark(chat_id, "（空白訊息）")
                return
            mm = [{"role": "user", "content": user_text}]
            resp = await call_openai_api(mm)
            reply = resp["choices"][0]["message"]["content"].strip()
            await send_message_to_lark(chat_id, reply)
            return

        # Other types not handled
        await send_message_to_lark(chat_id, "這個訊息類型暫不支援，請傳文字、圖片或檔案。")
    except Exception as e:
        logger.exception("Error handling message: %s", e)
        await send_message_to_lark(chat_id, f"抱歉，處理訊息時發生錯誤：{e}")

# --- Daily 08:00 summary of previous day for group chats ---
async def daily_summary_scheduler():
    while True:
        try:
            now_local = datetime.now(timezone(timedelta(hours=TIMEZONE_OFFSET)))
            next_run = now_local.replace(hour=8, minute=0, second=0, microsecond=0)
            if now_local >= next_run:
                next_run += timedelta(days=1)
            
            wait_seconds = (next_run - now_local).total_seconds()
            logger.info("Scheduler: Next summary run at %s (in %.2f hours)", next_run, wait_seconds / 3600)
            await asyncio.sleep(wait_seconds)

            run_time = datetime.now(timezone(timedelta(hours=TIMEZONE_OFFSET)))
            day_start = (run_time - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = (run_time - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)

            logger.info("Scheduler: Running daily summary for period %s to %s", day_start, day_end)
            
            # 從資料庫讀取資料，而非記憶體
            chats_to_summarize = get_chats_for_summary(day_start, day_end)

            for chat_id, messages in chats_to_summarize.items():
                if not messages:
                    continue
                content_lines = [f"{m['timestamp'].strftime('%H:%M')}: {m['text']}" for m in messages]
                combined = "\n".join(content_lines)
                prompt = "請將以下群組聊天記錄整理成摘要（繁體中文，條列式，含關鍵決策、待辦、未決問題）：\n\n" + combined[:15000]
                try:
                    resp = await call_openai_api([{"role": "user", "content": prompt}])
                    summary = resp["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    logger.exception("Summary generation failed for chat_id %s: %s", chat_id, e)
                    summary = "抱歉，今日聊天摘要生成失敗。"
                
                await send_message_to_lark(chat_id, f"昨日聊天摘要 ({day_start.strftime('%Y-%m-%d')}):\n{summary}")

            # 摘要完成後，在資料庫中標記，而非從記憶體清除
            if chats_to_summarize:
                mark_messages_as_summarized(day_start, day_end)

        except Exception as e:
            logger.exception("daily_summary_scheduler error: %s", e)
            await asyncio.sleep(60) # 如果發生未知錯誤，等待60秒後重試

@app.on_event("startup")
async def _startup():
    # 啟動時初始化資料庫
    init_db()
    asyncio.create_task(daily_summary_scheduler())

# --- Web server entry (optional for local run) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
