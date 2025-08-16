import os
import json
import logging
from typing import Tuple, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx

from zoneinfo import ZoneInfo
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("skycosmos-lark-bot")

APP_ID = os.getenv("APP_ID", "")
APP_SECRET = os.getenv("APP_SECRET", "")
LARK_BASE = "https://open.larksuite.com"

# Timezone for Taipei
TZ = ZoneInfo("Asia/Taipei")

app = FastAPI(title="Skycosmos Lark Bot", version="1.0.0")

async def get_tenant_access_token(app_id: str, app_secret: str) -> str:
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(
            f"{LARK_BASE}/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": app_id, "app_secret": app_secret},
        )
        r.raise_for_status()
        data = r.json()
        return data["tenant_access_token"]

async def send_text_to_chat(tenant_access_token: str, chat_id: str, text: str) -> None:
    url = f"{LARK_BASE}/open-apis/im/v1/messages?receive_id_type=chat_id"
    headers = {"Authorization": f"Bearer {tenant_access_token}", "Content-Type": "application/json"}
    payload = {
        "receive_id": chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": text}, ensure_ascii=False)
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()

async def download_message_resource(
    tenant_access_token: str,
    message_id: str,
    file_key: str,
    res_type: str = "file",
) -> Tuple[bytes, Optional[str], Optional[str]]:
    url = f"{LARK_BASE}/open-apis/im/v1/messages/{message_id}/resources/{file_key}"
    headers = {"Authorization": f"Bearer {tenant_access_token}"}
    params = {"type": res_type}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(url, headers=headers, params=params)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type")
        disp = r.headers.get("Content-Disposition", "")
        filename = None
        if "filename=" in disp:
            try:
                filename = disp.split("filename=", 1)[1].strip('"; ')
            except Exception:
                pass
        return r.content, filename, content_type

async def smart_parse_document(file_bytes: bytes, filename: str, mime: Optional[str]) -> str:
    size_kb = round(len(file_bytes) / 1024, 2)
    return f"已接收檔案：{filename}（{mime or 'unknown'}，約 {size_kb} KB）。\n（此處接你的實際解析流程…）"

@app.post("/webhook/lark")
async def lark_event(request: Request):
    body = await request.json()
    logger.info(f"Incoming event: {json.dumps(body, ensure_ascii=False)}")
    if body.get("type") == "url_verification" and "challenge" in body:
        return JSONResponse({"challenge": body["challenge"]})
    event = body.get("event", {})
    event_type = event.get("type")
    if event_type == "message":
        message = event.get("message", {})
        msg_id = message.get("message_id")
        msg_type = message.get("message_type")
        content_raw = message.get("content") or "{}"
        content = json.loads(content_raw)
        chat_id = message.get("chat_id")
        try:
            tat = await get_tenant_access_token(APP_ID, APP_SECRET)
        except httpx.HTTPError:
            logger.exception("取得 tenant_access_token 失敗")
            return JSONResponse({"code": 0})
        if msg_type == "text":
            text = content.get("text", "")
            await send_text_to_chat(tat, chat_id, f"已收到文字：{text}")
            return JSONResponse({"code": 0})
        if msg_type == "file":
            file_key = content.get("file_key")
            user_filename = content.get("file_name")
            if not (msg_id and file_key):
                await send_text_to_chat(tat, chat_id, "收到檔案但缺少必要識別（message_id 或 file_key）。")
                return JSONResponse({"code": 0})
            try:
                file_bytes, srv_name, mime = await download_message_resource(
                    tat, msg_id, file_key, res_type="file"
                )
                filename = srv_name or user_filename or f"{file_key}.bin"
                summary = await smart_parse_document(file_bytes, filename, mime)
                await send_text_to_chat(tat, chat_id, summary)
            except httpx.HTTPError:
                logger.exception("檔案下載失敗")
                await send_text_to_chat(tat, chat_id, "下載檔案失敗，請稍後重試。")
            return JSONResponse({"code": 0})
        if msg_type == "image":
            image_key = content.get("image_key")
            if not (msg_id and image_key):
                await send_text_to_chat(tat, chat_id, "收到圖片但缺少必要識別（message_id 或 image_key）。")
                return JSONResponse({"code": 0})
            try:
                img_bytes, srv_name, mime = await download_message_resource(
                    tat, msg_id, image_key, res_type="image"
                )
                name = srv_name or f"{image_key}.jpg"
                await send_text_to_chat(tat, chat_id, f"已收到圖片：{name}（{mime or 'unknown'}）。")
            except httpx.HTTPError:
                logger.exception("圖片下載失敗")
                await send_text_to_chat(tat, chat_id, "下載圖片失敗，請稍後重試。")
            return JSONResponse({"code": 0})
        await send_text_to_chat(tat, chat_id, f"收到訊息類型：{msg_type}（暫未支援）。")
        return JSONResponse({"code": 0})
    return JSONResponse({"code": 0})

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

scheduler = AsyncIOScheduler(timezone=TZ)

async def morning_summary_job():
    CHAT_ID = os.getenv("SUMMARY_CHAT_ID")
    if not CHAT_ID:
        logger.warning("未設定 SUMMARY_CHAT_ID，略過每日摘要訊息。")
        return
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    text = (
        f"({today})\n"
        "### 群組聊天記錄摘要\n\n"
        "#### 關鍵決策\n- （示例）...\n\n"
        "#### 待辦事項\n- （示例）...\n\n"
        "#### 未決問題\n- （示例）...\n"
    )
    try:
        tat = await get_tenant_access_token(APP_ID, APP_SECRET)
        await send_text_to_chat(tat, CHAT_ID, text)
        logger.info("已發送每日摘要")
    except Exception:
        logger.exception("每日摘要發送失敗")


def setup_scheduler():
    scheduler.add_job(
        morning_summary_job,
        CronTrigger(hour=8, minute=0),
        id="daily_morning_summary",
        max_instances=1,
        coalesce=True
    )
    scheduler.start()
    logger.info("APScheduler started (Asia/Taipei 08:00 daily)")

setup_scheduler()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=True)
