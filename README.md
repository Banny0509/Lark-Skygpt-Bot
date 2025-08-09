# Lark‑SkyGPT Bot (FastAPI)

**Language / 語言**: [中文（繁體）](#zh-hant) | [English](#en)

---

<a id="zh-hant"></a>

# 中文（繁體）

將 OpenAI ChatGPT 集成至 **Lark／飛書**，並部署在 **Railway**。本專案已依現有程式與環境變數完成配置：**FastAPI + Uvicorn + Gunicorn + httpx + python‑dotenv**。

> 典型場景：私訊自動回覆、群組需 @ 機器人才觸發、統一呼叫 OpenAI 產生回覆，再透過 Lark 訊息 API 回傳。

## 目錄
- [功能概覽](#功能概覽)
- [目錄結構](#目錄結構)
- [運作機制](#運作機制)
- [環境變數](#環境變數)
- [本地開發](#本地開發)
- [部署到 Railway](#部署到-railway)
- [在 Lark 開發者平台配置](#在-lark-開發者平台配置)
- [介面說明](#介面說明)
- [日誌與排錯](#日誌與排錯)
- [常見問題](#常見問題)
- [Roadmap](#roadmap-zh)

---

## 功能概覽

- **私訊（p2p）自動觸發**：使用者直接私訊機器人即回覆。
- **群組需 @ 才觸發**：只有在群組訊息中 **@機器人** 才會回應，避免干擾。
- **文字訊息支援**：目前僅處理 `text` 類型（其他類型忽略）。
- **工具調用（function calling）示例**：內建 `get_current_time_and_date`；當詢問「今天幾號／幾點／星期幾」時，模型會自動呼叫此工具再回覆。
- **健康檢查**：`GET /health` 回傳 200，用於雲端健康檢測。
- **最小依賴**：`fastapi、uvicorn、gunicorn、python-dotenv、httpx`。

## 目錄結構

```
Lark-Skygpt-Bot-main/
└─ Lark-Skygpt-Bot-main/
   ├─ main.py                 # FastAPI 應用、Webhook、Lark 與 OpenAI 的調用邏輯
   ├─ requirements.txt        # 依賴清單
   ├─ Procfile                # Railway / Gunicorn 啟動命令
   ├─ .env                    # 本地開發的環境變數（勿提交到 Git）
   └─ README.md               # 專案說明（可替換為本文檔）
```

## 運作機制

- **Webhook**：Lark 事件訂閱指向 `POST /webhook`。首次驗證會回傳 `challenge`。
- **事件處理**：收到 `im.message.receive_v1`：
  - 私訊：直接觸發。
  - 群組：僅當訊息包含對機器人的 **@提及** 時觸發。
  - 僅處理 `text` 訊息；會先移除 `@機器人` 文本後再作為使用者問題。
- **OpenAI 呼叫**：
  - 預設模型：`gpt-4-turbo`（可於 `call_openai_api` 調整）。
  - 若模型判斷需要「當前日期／時間」，會自動呼叫 `get_current_time_and_date` 工具。
- **回傳訊息**：使用 Lark `im/v1/messages?receive_id_type=chat_id` API 回傳文字到原會話。

## 環境變數

於 **Railway** 或其他雲端以變數設定；本地開發則放在專案根目錄 `.env`：

| 變數名 | 說明 |
|---|---|
| `APP_ID` | Lark 應用 **App ID** |
| `APP_SECRET` | Lark 應用 **App Secret** |
| `VERIFICATION_TOKEN` | Lark 事件訂閱 **Verification Token** |
| `OPENAI_API_KEY` | OpenAI API Key |

> 程式會依 `RAILWAY_ENVIRONMENT` 是否存在決定：本地自動載入 `.env`，雲端直接讀平台變數。

範例（`.env`）：
```env
APP_ID=your_lark_app_id
APP_SECRET=your_lark_app_secret
VERIFICATION_TOKEN=your_lark_verification_token
OPENAI_API_KEY=your_openai_api_key
```

## 本地開發

1. 安裝依賴
   ```bash
   pip install -r requirements.txt
   ```
2. 設定 `.env`（見上）。
3. 啟動服務
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. 本地聯調：以 `ngrok` 或 `cloudflared` 暴露公網，將 Lark 事件訂閱暫指向 `https://你的隧道域名/webhook`。

## 部署到 Railway

1. 建立專案並連結儲存庫（或直接上傳程式）。
2. 於 **Settings → Variables** 新增上方四個環境變數。
3. 啟動命令已在 `Procfile` 指定：
   ```
   web: gunicorn -w 2 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT
   ```
4. 分配公網網域：在 Railway 網域後加上 `/webhook` 作為 Lark 事件訂閱 URL。
5. 健康檢查（可選）：指向 `/health`。

## 在 Lark 開發者平台配置

1. 建立應用並取得 **App ID / App Secret**。
2. 權限：開啟「收發訊息、事件訂閱」等必要權限（以官方頁面為準）。
3. 事件訂閱（Event Subscriptions）：
   - **Request URL**：`https://你的網域/webhook`
   - **Verification Token**：填寫並與環境變數一致。
   - **訂閱事件**：勾選 **`im.message.receive_v1`**（接收訊息事件）。
4. 安裝至企業／群組，於群內 @ 機器人測試。

## 介面說明

### `GET /health`
- **用途**：健康檢查
- **回應**：`200 OK`，JSON：`{"status":"ok"}`（或於日誌可見正常輸出）

### `POST /webhook`
- **用途**：接收 Lark 事件
- **握手**：若 body 含 `challenge`，回傳 `{ "challenge": "<challenge>" }`。
- **訊息處理**：僅對私訊或群組 @ 觸發的 **文字** 訊息進行處理，再呼叫 OpenAI 產生回覆並回傳。

## 日誌與排錯

- 預設日誌等級 **INFO**。部署後可於平台日誌查看。
- 常見錯誤：
  - **401/403**：`APP_ID / APP_SECRET / VERIFICATION_TOKEN` 不一致或無效。
  - **無法發訊息**：應用未安裝至目標會話、缺少權限、或 `chat_id` 錯誤。
  - **OpenAI 錯誤**：檢查 `OPENAI_API_KEY`、模型名稱與額度。

## 常見問題

**Q：群組為何沒有回應？**  
A：需 **@機器人** 才會觸發；或檢查事件權限與訂閱是否已開啟。

**Q：可否支援圖片／檔案？**  
A：目前僅處理 `text`。可於 `handle_message_receive` 擴充分支並接入相應 API。

**Q：如何固定使用繁體中文回覆？**  
A：在 `messages` 的 `system` 提示語加入「請一律使用繁體中文回覆」。若要在日誌中顯示非轉義中文，將回傳內容改為：
```python
json.dumps({"text": text}, ensure_ascii=False)
```

## Roadmap {#roadmap-zh}

- [ ] 加入「每日 12:00 發送 Base 變動報告、每週五 18:00 發送週報」的排程（如 APScheduler / Actions + API）。
- [ ] 支援富文本／卡片訊息。
- [ ] 對話記憶與持久化。
- [ ] 測試與 CI。

---

<a id="en"></a>

# English

Integrate OpenAI ChatGPT into **Lark (Feishu)** and deploy on **Railway**. The project matches your current stack: **FastAPI + Uvicorn + Gunicorn + httpx + python‑dotenv**.

> Typical use cases: auto reply in DMs, reply-on-mention in group chats, unify responses via OpenAI, then send back through Lark Message API.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Environment Variables](#environment-variables)
- [Local Development](#local-development)
- [Deploy to Railway](#deploy-to-railway)
- [Lark Developer Console Settings](#lark-developer-console-settings)
- [API Endpoints](#api-endpoints)
- [Logging & Troubleshooting](#logging--troubleshooting)
- [FAQ](#faq)
- [Roadmap](#roadmap)

---

## Features

- **Auto-trigger in DMs (p2p)**.
- **Group chats trigger on @mention** only, to reduce noise.
- **Text messages supported** (`text` only at the moment).
- **Function calling example**: built-in `get_current_time_and_date`; when users ask about “today/time/weekday”, the model calls the tool first.
- **Health check**: `GET /health` returns `200`.
- **Lean dependencies**: `fastapi`, `uvicorn`, `gunicorn`, `python-dotenv`, `httpx`.

## Project Structure

```
Lark-Skygpt-Bot-main/
└─ Lark-Skygpt-Bot-main/
   ├─ main.py                 # FastAPI app, webhook, Lark & OpenAI integrations
   ├─ requirements.txt        # Dependencies
   ├─ Procfile                # Railway / Gunicorn start command
   ├─ .env                    # Local-only environment variables (do not commit)
   └─ README.md               # Project doc (you can replace it with this file)
```

## How It Works

- **Webhook**: Lark event subscription points to `POST /webhook`. Initial verification echoes `challenge`.
- **Event handling** on `im.message.receive_v1`:
  - DMs: always trigger.
  - Groups: trigger only when the bot is **mentioned**.
  - Only `text` is processed; the `@bot` mention text is stripped before sending to OpenAI.
- **OpenAI**:
  - Default model: `gpt-4-turbo` (adjust in `call_openai_api`).
  - If the model needs “current date/time”, it calls `get_current_time_and_date`.
- **Reply**: send back via `im/v1/messages?receive_id_type=chat_id`.

## Environment Variables

Configure in your cloud platform (e.g., **Railway**). For local dev, put them in `.env`:

| Name | Description |
|---|---|
| `APP_ID` | Lark **App ID** |
| `APP_SECRET` | Lark **App Secret** |
| `VERIFICATION_TOKEN` | Event subscription **Verification Token** |
| `OPENAI_API_KEY` | OpenAI API key |

> The app auto-loads `.env` locally and reads platform variables in production (checks `RAILWAY_ENVIRONMENT`).

Example (`.env`):
```env
APP_ID=your_lark_app_id
APP_SECRET=your_lark_app_secret
VERIFICATION_TOKEN=your_lark_verification_token
OPENAI_API_KEY=your_openai_api_key
```

## Local Development

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Create `.env` as above.
3. Run the server
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Expose tunnel (`ngrok`/`cloudflared`) and temporarily set Lark event subscription to `https://your-tunnel/webhook`.

## Deploy to Railway

1. Create a project and connect your repo (or upload code).
2. Add the four variables under **Settings → Variables**.
3. Start command is defined in `Procfile`:
   ```
   web: gunicorn -w 2 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT
   ```
4. Use the Railway domain with `/webhook` as the Lark subscription URL.
5. (Optional) Health check targets `/health`.

## Lark Developer Console Settings

1. Create an app and note **App ID / App Secret**.
2. Permissions: enable messaging & event subscriptions as required by Lark.
3. Event Subscriptions:
   - **Request URL**: `https://your-domain/webhook`
   - **Verification Token**: must match your env var
   - **Subscribed events**: **`im.message.receive_v1`**
4. Install to your tenant/groups and test by mentioning the bot.

## API Endpoints

### `GET /health`
- **Use**: health checks
- **Return**: `200 OK`, JSON `{ "status": "ok" }`

### `POST /webhook`
- **Use**: receive Lark events
- **Handshake**: echo `{ "challenge": "<challenge>" }` if present.
- **Processing**: handle **text** in DMs or mention-triggered group chats, call OpenAI, then reply.

## Logging & Troubleshooting

- Default log level: **INFO**.
- Common issues:
  - **401/403**: invalid or mismatched `APP_ID / APP_SECRET / VERIFICATION_TOKEN`.
  - **Cannot send message**: app not installed in the target chat, missing permission, or wrong `chat_id`.
  - **OpenAI errors**: verify `OPENAI_API_KEY`, model name, and quota.

## FAQ

**Q: Why no response in group chats?**  
A: You must **@mention** the bot; also check permissions and event subscriptions.

**Q: Can it handle images/files?**  
A: Currently `text` only. Extend branches in `handle_message_receive` to support more types.

**Q: Force Traditional Chinese outputs?**  
A: Add a system instruction like “Always reply in Traditional Chinese.” For non-escaped logs, use:
```python
json.dumps({"text": text}, ensure_ascii=False)
```

## Roadmap

- [ ] Scheduler to send daily Base deltas at 12:00 and weekly report at 18:00 on Fridays.
- [ ] Rich text/cards.
- [ ] Conversation memory & persistence.
- [ ] Tests & CI.
