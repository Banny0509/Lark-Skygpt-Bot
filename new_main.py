# main.py — Lark SkyGPT Bot (per-chat daily summary)
# 功能：
#  - Webhook：/webhook 與 /webhook/lark（兩條路徑皆可）
#  - 正確下載端點：/im/v1/messages/{message_id}/resources/{key}?type=file|image
#  - 每天 08:00 (Asia/Taipei) 對「昨天有聊天」的每個群，各自發該群摘要（DB 鎖防重覆）
#  - 問答含時間感知、圖片理解、文件讀取與摘要
#  - 單 worker 友善（建議 Gunicorn -w 1）

import os
import io
import re
import json
import base64
import logging
import mimetypes
import asyncio
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from zoneinfo import ZoneInfo

# 可選依賴（如未安裝將自動降級）
try:
    import aiosqlite
except Exception:
    aiosqlite = None

try:
    from pypdf import PdfReader
    HAVE_PYPDF = True
except Exception:
    HAVE_PYPDF = False

try:
    import docx2txt
    HAVE_DOCX = True
except Exception:
    HAVE_DOCX = False

# OpenAI（可選）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
USE_OPENAI = bool(OPENAI_API_KEY)
if USE_OPENAI:
    try:
        from openai import OpenAI
        oai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        USE_OPENAI = False
        oai_client = None

APP_ID = os.getenv("APP_ID", "").strip()
APP_SECRET = os.getenv("APP_SECRET", "").strip()
LARK_BASE = "https://open.larksuite.com"
TIMEZONE = os.getenv("TZ", "Asia/Taipei")
TZ = ZoneInfo(TIMEZONE)
DB_PATH = os.getenv("DB_PATH", "data.db")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("skygpt")
app = FastAPI(title="Lark SkyGPT Bot")

# ====== Token 快取 ======
_token_cache: Dict[str, Any] = {"tenant_access_token": None, "expire_at": 0}
async def get_tenant_access_token() -> str:
    now = int(datetime.now(tz=timezone.utc).timestamp())
    if _token_cache["tenant_access_token"] and now < _token_cache["expire_at"] - 60:
        return _token_cache["tenant_access_token"]
    url = f"{LARK_BASE}/open-apis/auth/v3/tenant_access_token/internal"
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(url, json={"app_id": APP_ID, "app_secret": APP_SECRET})
        r.raise_for_status()
        data = r.json()
    tat = data.get("tenant_access_token")
    expire = int(data.get("expire", 3600))
    _token_cache["tenant_access_token"] = tat
    _token_cache["expire_at"] = now + expire
    return tat

# ====== Lark 發送文字 ======
async def send_text_to_chat(chat_id: str, text: str) -> None:
    tat = await get_tenant_access_token()
    api = f"{LARK_BASE}/open-apis/im/v1/messages?receive_id_type=chat_id"
    payload = {"receive_id": chat_id, "content": json.dumps({"text": text}, ensure_ascii=False), "msg_type": "text"}
    headers = {"Authorization": f"Bearer {tat}"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(api, headers=headers, json=payload)
        if r.status_code >= 400:
            logger.error("send_text_to_chat failed: %s %s", r.status_code, r.text)

# ====== 正確的下載端點（帶 type）======
async def download_message_resource(
    message_id: str,
    key: str,
    res_type: str = "file",  # "file" 或 "image"
) -> Tuple[bytes, Optional[str], Optional[str]]:
    """
    GET /open-apis/im/v1/messages/:message_id/resources/:key?type=file|image
    回傳 (bytes, filename, content_type)
    """
    tat = await get_tenant_access_token()
    url = f"{LARK_BASE}/open-apis/im/v1/messages/{message_id}/resources/{key}"
    headers = {"Authorization": f"Bearer {tat}"}
    params = {"type": res_type}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(url, headers=headers, params=params)
        r.raise_for_status()
        ct = r.headers.get("Content-Type")
        disp = r.headers.get("Content-Disposition", "")
        name = None
        if "filename=" in disp:
            try:
                name = disp.split("filename=", 1)[1].strip('"; ')
            except Exception:
                pass
        return r.content, name, ct

# ====== 以時間範圍拉群組訊息（用於摘要補齊）======
async def list_chat_messages_between(chat_id: str, start_ms: int, end_ms: int, page_size: int = 100) -> List[Dict[str, Any]]:
    tat = await get_tenant_access_token()
    headers = {"Authorization": f"Bearer {tat}"}
    api = f"{LARK_BASE}/open-apis/im/v1/messages"
    params = {
        "container_id_type": "chat",
        "container_id": chat_id,
        "start_time": str(start_ms),
        "end_time": str(end_ms),
        "page_size": str(page_size),
    }
    items: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            r = await client.get(api, headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
            page = data.get("data", {}).get("items", []) or []
            items.extend(page)
            token = data.get("data", {}).get("page_token")
            if not token:
                break
            params["page_token"] = token
    return items

# ====== SQLite（訊息存檔與每日鎖）======
async def ensure_db():
    if not aiosqlite:
        logger.warning("aiosqlite 未安裝，將無法做 per-chat 歷史與鎖。")
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            message_id TEXT,
            sender_id TEXT,
            ts_ms INTEGER,
            msg_type TEXT,
            text TEXT,
            file_key TEXT,
            image_key TEXT
        )""")
        await db.execute("""
        CREATE TABLE IF NOT EXISTS summary_lock(
            summary_date TEXT,   -- YYYY-MM-DD (local)
            chat_id TEXT,
            PRIMARY KEY(summary_date, chat_id)
        )""")
        await db.commit()

async def db_insert_message(row: Dict[str, Any]):
    if not aiosqlite:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
        INSERT INTO messages (chat_id, message_id, sender_id, ts_ms, msg_type, text, file_key, image_key)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", (
            row.get("chat_id"), row.get("message_id"), row.get("sender_id"),
            row.get("ts_ms"), row.get("msg_type"), row.get("text"),
            row.get("file_key"), row.get("image_key"),
        ))
        await db.commit()

async def acquire_summary_lock(date_str: str, chat_id: str) -> bool:
    if not aiosqlite:
        return True
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("INSERT INTO summary_lock (summary_date, chat_id) VALUES (?, ?)", (date_str, chat_id))
            await db.commit()
            return True
    except Exception:
        return False

async def db_list_chat_ids_with_messages_between(start_ms: int, end_ms: int) -> List[str]:
    if not aiosqlite:
        return []
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT DISTINCT chat_id FROM messages WHERE ts_ms >= ? AND ts_ms < ?",
            (start_ms, end_ms)
        )
        rows = await cur.fetchall()
    return [r[0] for r in rows if r and r[0]]

# ====== 時間工具 ======
def now_local() -> datetime:
    return datetime.now(TZ)

def yesterday_range_local() -> Tuple[datetime, datetime]:
    today = now_local().date()
    y = today - timedelta(days=1)
    start = datetime(y.year, y.month, y.day, 0, 0, 0, tzinfo=TZ)
    end = start + timedelta(days=1)
    return start, end

def to_epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

# ====== 內容解析 ======
def parse_msg_item_basic(item: Dict[str, Any]) -> Dict[str, Any]:
    msg = item.get("message") or item
    message_id = msg.get("message_id") or ""
    chat_id = msg.get("chat_id") or item.get("chat_id")
    msg_type = msg.get("message_type") or msg.get("msg_type")
    sender = msg.get("sender") or {}
    sender_id = sender.get("sender_id") or sender.get("open_id") or ""

    ts_ms = 0
    if "create_time" in msg:
        try:
            ts_ms = int(msg["create_time"])
        except Exception:
            pass

    content_str = msg.get("content") or "{}"
    try:
        content = json.loads(content_str)
    except Exception:
        content = {}

    text = None; file_key = None; image_key = None
    if msg_type == "text":
        text = content.get("text")
    elif msg_type == "post":
        text = re.sub(r"<.*?>", "", content_str)
    elif msg_type == "file":
        file_key = content.get("file_key")
        text = content.get("file_name")
    elif msg_type == "image":
        image_key = content.get("image_key")
    else:
        text = content_str

    return {
        "chat_id": chat_id, "message_id": message_id, "msg_type": msg_type,
        "sender_id": sender_id, "ts_ms": ts_ms, "text": text,
        "file_key": file_key, "image_key": image_key, "raw": msg,
    }

# ====== 檔案與圖片解析 ======
def guess_filename(default_name: str, content_type: Optional[str], header_name: Optional[str]) -> str:
    if header_name:
        return header_name
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ""
        if ext and not default_name.endswith(ext):
            return default_name + ext
    return default_name

def safe_decode_text(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "big5", "gbk", "latin1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")

def extract_text_from_pdf(data: bytes) -> str:
    if not HAVE_PYPDF:
        return "[PDF 已接收，但伺服器未安裝 pypdf]"
    try:
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages[:20]:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                pass
        text = "\n".join(parts).strip()
        return text or "[PDF 無可抽取文字]"
    except Exception:
        return "[PDF 解析失敗]"

def extract_text_from_docx(data: bytes) -> str:
    if not HAVE_DOCX:
        return "[DOCX 已接收，但伺服器未安裝 docx2txt]"
    try:
        import tempfile, os as _os
        with tempfile.TemporaryDirectory() as td:
            p = _os.path.join(td, "tmp.docx")
            with open(p, "wb") as f:
                f.write(data)
            text = docx2txt.process(p) or ""
            return text.strip() or "[DOCX 無可抽取文字]"
    except Exception:
        return "[DOCX 解析失敗]"

def extract_text_generic(data: bytes, filename: str, content_type: Optional[str]) -> str:
    name = filename.lower()
    if name.endswith(".pdf") or (content_type and "pdf" in content_type):
        return extract_text_from_pdf(data)
    if name.endswith(".docx") or (content_type and "officedocument.wordprocessingml" in content_type):
        return extract_text_from_docx(data)
    if name.endswith(".csv") or (content_type and "csv" in content_type):
        try:
            return "\n".join(safe_decode_text(data).splitlines()[:50])
        except Exception:
            return "[CSV 解析失敗]"
    if name.endswith(".txt") or (content_type and "text/plain" in content_type):
        return safe_decode_text(data)
    try:
        return safe_decode_text(data)
    except Exception:
        return f"[{filename}（{len(data)} bytes）]"

# ====== OpenAI 幫助 ======
async def openai_text_completion(prompt: str, sys: str = "You are a helpful assistant.") -> str:
    if not USE_OPENAI:
        return prompt[:8000]
    try:
        resp = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return prompt[:8000]

async def openai_vision_describe(image_bytes: bytes, extra_prompt: str = "") -> str:
    if not USE_OPENAI:
        return "[已接收圖片]"
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        messages = [
            {"role": "system", "content": "You are an expert image analyst."},
            {"role": "user", "content": [
                {"type": "text", "text": extra_prompt or "請描述這張圖片的重點。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}" }},
            ]}
        ]
        resp = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "[圖片解析失敗]"

# ====== 摘要 ======
SUMMARY_PROMPT = """你是資深會議/群組助理。請將下列聊天訊息整理成摘要（繁體中文）：
輸出格式：
(YYYY-MM-DD):
### 群組聊天記錄摘要

#### 關鍵決策
- ...

#### 待辦事項
- ...

#### 未決問題
- ...

僅總結「昨天（本地時區）」00:00–24:00 的內容，簡潔具體。"""

def render_yesterday_title(dt: datetime) -> str:
    y = dt.date() - timedelta(days=1)
    return f"({y.isoformat()}):"

async def summarize_messages_text(snippets: List[str]) -> str:
    text = "\n".join(f"- {s}" for s in snippets)
    if USE_OPENAI:
        prompt = f"{SUMMARY_PROMPT}\n\n---\n聊天摘錄：\n{text[:12000]}"
        return await openai_text_completion(prompt)
    return f"""{render_yesterday_title(now_local())}
### 群組聊天記錄摘要

#### 關鍵決策
- （未配置 OpenAI；共 {len(snippets)} 則訊息）

#### 待辦事項
- 依訊息整理行動項

#### 未決問題
- （如有）"""

# ====== Webhook 入口 ======
@app.post("/webhook")
async def webhook_alias(request: Request):
    return await lark_event(request)

@app.post("/webhook/lark")
async def lark_event(request: Request):
    body = await request.json()
    if "challenge" in body:
        return JSONResponse({"challenge": body["challenge"]})
    event = body.get("event", {}) or {}
    header = body.get("header", {}) or {}
    etype = header.get("event_type") or event.get("type")
    if etype and "message" in etype:
        await handle_message_event(event)
    return JSONResponse({"code": 0})

# ====== 訊息處理 ======
async def handle_message_event(event: Dict[str, Any]):
    msg = event.get("message") or {}
    chat_id = msg.get("chat_id")
    message_id = msg.get("message_id")
    msg_type = msg.get("message_type")
    sender = event.get("sender", {}) or {}
    sender_open_id = (sender.get("sender_id") or {}).get("open_id") or ""
    create_ms = 0
    try:
        create_ms = int(msg.get("create_time") or "0")
    except Exception:
        pass

    content_str = msg.get("content") or "{}"
    try:
        content = json.loads(content_str)
    except Exception:
        content = {}

    # 入庫
    try:
        await db_insert_message({
            "chat_id": chat_id,
            "message_id": message_id,
            "sender_id": sender_open_id,
            "ts_ms": create_ms,
            "msg_type": msg_type,
            "text": content.get("text") if msg_type == "text" else None,
            "file_key": content.get("file_key") if msg_type == "file" else None,
            "image_key": content.get("image_key") if msg_type == "image" else None,
        })
    except Exception as e:
        logger.debug("db insert skipped: %s", e)

    # 指令與一般應答
    if msg_type == "text":
        text = (content.get("text") or "").strip()
        if not text:
            return
        if text.startswith("/help"):
            await send_text_to_chat(chat_id,
                "指令：\n"
                "/time 現在時間\n"
                "/date 今日日期\n"
                "/summary 立即彙整昨天摘要（只對本群）\n"
                "(或直接提問)")
            return
        if text.startswith("/time"):
            await send_text_to_chat(chat_id, now_local().strftime("現在時間：%Y-%m-%d %H:%M:%S %Z"))
            return
        if text.startswith("/date"):
            await send_text_to_chat(chat_id, now_local().strftime("今日日期：%Y-%m-%d（%A）"))
            return
        if text.startswith("/summary"):
            await summarize_for_single_chat(chat_id)
            return
        # 一般問答
        if USE_OPENAI:
            prompt = f"現在本地時間是 {now_local().strftime('%Y-%m-%d %H:%M:%S %Z')}。\n使用繁體中文回答：\n\n使用者：{text}"
            out = await openai_text_completion(prompt, sys="You are a helpful assistant in Traditional Chinese.")
            await send_text_to_chat(chat_id, out)
        else:
            await send_text_to_chat(chat_id, f"(無 LLM) 你說：{text}\n現在：{now_local().strftime('%Y-%m-%d %H:%M:%S %Z')}")
        return

    if msg_type == "image":
        image_key = content.get("image_key")
        if not image_key:
            await send_text_to_chat(chat_id, "收到圖片，但缺少 image_key。")
            return
        try:
            img_bytes, _, _ = await download_message_resource(message_id, image_key, "image")
            desc = await openai_vision_describe(img_bytes)
            await send_text_to_chat(chat_id, desc)
        except Exception as e:
            logger.exception("image fail: %s", e)
            await send_text_to_chat(chat_id, "圖片下載/解析失敗，請稍後再試。")
        return

    if msg_type == "file":
        file_key = content.get("file_key")
        file_name = content.get("file_name") or "file"
        if not file_key:
            await send_text_to_chat(chat_id, "收到檔案，但缺少 file_key。")
            return
        try:
            data, header_name, content_type = await download_message_resource(message_id, file_key, "file")
            fname = guess_filename(file_name, content_type, header_name)
            text = extract_text_generic(data, fname, content_type)
            if USE_OPENAI:
                prompt = f"以下是使用者上傳文件「{fname}」的內容摘錄，請以繁體中文摘要重點與待辦：\n\n{text[:12000]}"
                out = await openai_text_completion(prompt)
            else:
                out = f"(無 LLM)\n檔名：{fname}\n前 400 字：\n{text[:400]}"
            await send_text_to_chat(chat_id, out)
        except Exception as e:
            logger.exception("file fail: %s", e)
            await send_text_to_chat(chat_id, "檔案下載/解析失敗，請稍後再試。")
        return

# ====== 每日摘要：逐群處理 ======
def _yesterday_title() -> str:
    y = now_local().date() - timedelta(days=1)
    return f"({y.isoformat()}):"

async def summarize_for_single_chat(chat_id: str):
    start_dt, end_dt = yesterday_range_local()
    start_ms, end_ms = to_epoch_ms(start_dt), to_epoch_ms(end_dt)
    date_tag = (start_dt.date()).isoformat()
    if not await acquire_summary_lock(date_tag, chat_id):
        return
    try:
        items = await list_chat_messages_between(chat_id, start_ms, end_ms)
    except Exception as e:
        logger.warning("list by API failed, fallback DB: %s", e)
        items = []

    snippets: List[str] = []
    if items:
        for it in items:
            meta = parse_msg_item_basic(it)
            if meta["msg_type"] == "text" and meta.get("text"):
                clean = re.sub(r"<.*?>", "", meta["text"]).strip()
                if clean:
                    snippets.append(clean[:300])
    elif aiosqlite:
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute(
                "SELECT text FROM messages WHERE chat_id=? AND ts_ms >= ? AND ts_ms < ? AND msg_type='text'",
                (chat_id, start_ms, end_ms)
            )
            rows = await cur.fetchall()
            for (t,) in rows:
                if t:
                    clean = re.sub(r"<.*?>", "", t).strip()
                    if clean:
                        snippets.append(clean[:300])

    if not snippets:
        await send_text_to_chat(chat_id, f"{_yesterday_title()}\n（昨天沒有可摘要的文字訊息）")
        return
    summary = await summarize_messages_text(snippets)
    await send_text_to_chat(chat_id, summary)

async def run_daily_summary_per_chat():
    start_dt, end_dt = yesterday_range_local()
    start_ms, end_ms = to_epoch_ms(start_dt), to_epoch_ms(end_dt)
    if not aiosqlite:
        logger.warning("aiosqlite 不可用，無法自動列出群組；跳過每日摘要。")
        return
    chat_ids = await db_list_chat_ids_with_messages_between(start_ms, end_ms)
    if not chat_ids:
        logger.info("昨天沒有發現有聊天的群組，略過摘要。")
        return
    for cid in chat_ids:
        try:
            await summarize_for_single_chat(cid)
        except Exception as e:
            logger.exception("summary for chat %s failed: %s", cid, e)

# ====== APScheduler（每天 08:00）======
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

scheduler: Optional[AsyncIOScheduler] = None
def start_scheduler():
    global scheduler
    if scheduler:
        return
    scheduler = AsyncIOScheduler(timezone=TZ)
    scheduler.add_job(
        lambda: asyncio.create_task(run_daily_summary_per_chat()),
        CronTrigger(hour=8, minute=0, second=0, timezone=TZ),
        id="daily_summary_0800_per_chat",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )
    scheduler.start()
    logger.info("Scheduler started (08:00 %s)", TIMEZONE)

# ====== 健康檢查 ======
@app.get("/")
async def root_ok():
    return PlainTextResponse("ok")

@app.get("/healthz")
async def healthz():
    return JSONResponse({
        "status": "ok",
        "now": now_local().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "tz": TIMEZONE,
        "openai": bool(USE_OPENAI),
    })

# ====== 啟動 ======
@app.on_event("startup")
async def on_startup():
    if not APP_ID or not APP_SECRET:
        logger.warning("APP_ID/APP_SECRET 未設定，Lark API 將無法運作。")
    if aiosqlite:
        await ensure_db()
    else:
        logger.warning("未安裝 aiosqlite，per-chat 摘要將無法運作。")
    start_scheduler()
    logger.info("SkyGPT ready.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
