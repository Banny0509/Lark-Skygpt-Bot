# Lark-SkyGPT Bot (FastAPI)

**Language / 語言**: [中文 (繁體)](#zh-hant) | [English](#en)

---

<a id="zh-hant"></a>

## 中文 (繁體)

將 OpenAI ChatGPT 集成至 **Lark／飛書**，並部署至 **Railway**。本專案現在已具備多模態處理能力，可以接收文本、圖片與檔案（例如 Excel、Word、PowerPoint、PDF），解析其中的內容並透過 OpenAI 生成回復或摘要。當收到圖片或圖表時，會將視覺資料轉成適合模型分析的格式，再送交模型推理。

### 功能概要
- 私訊或群組聊天機器人：使用者可以在私聊或群組裡與機器人互動。
- **多模態支援**：除了處理文字訊息外，現在還可處理圖片以及常用辦公檔案（Excel、Word、PowerPoint、PDF 等）。這些檔案會經過解析後交給模型生成摘要或回答。
- **函數購命（function calling）**：內建範例函數 `get_current_time_and_date`，當詢問「今天幾點／模操／星期關事」時，模型會自動呼叫此工具來回要。
- **最小依譜更新**：本專案依譜 `fastapi、uvicorn、gunicorn、python-dotenv、httpx`，並新增多模態相關套件：`openpyxl、python-docx、python-pptx、pdfplumber、pillow`。


### 運作模組
- 機器人監聽 Lark webhook，接收事件。
- 根據事件類型處理不同訊息：
  - `text`：直接將文字傳給 OpenAI 生成回復。
  - `image`：下載圖片、轉成 Base64，組成多模態 payload，送給 OpenAI 進行視覺理解後回復。
  - `file`：根據檔案類型（Excel／Word／PPT／PDF），使用相應解析庫讀取內容，再傳給 OpenAI 生成回復或摘要。
- 將回應回傳至 Lark 用戶。

### API
- `POST /webhook`：接收 Lark 事件（text、image、file）並處理訊息。
- `GET /health`：健康檢查。

### FAQ
**可以處理圖片或檔案嗎？** 可以！現在支援圖片訊息以及 Excel、Word、PowerPoint、PDF 等文件，檔案內容會經過解析後用於生成摘要或回答。

---

<a id="en"></a>

## English

Integrate OpenAI's ChatGPT into **Lark** and deploy on **Railway**. This project now supports multi‑modal input — text, images and common office documents (Excel, Word, PowerPoint, PDF). The bot will parse file contents or extract information from images and then ask OpenAI to summarize or answer questions.

### Features
- **Chatbot for private/group chats**: Users can interact with the bot either in direct chats or group chats.
- **Multi‑modal support**: Beyond plain text, the bot can handle images and files (Excel, Word, PowerPoint, PDF). Files are parsed and the extracted content is sent to the model for analysis.
- **Function calling**: Includes a sample function `get_current_time_and_date`. When the user asks about the current date/time, the model will call this function automatically.
- **Updated lean dependencies**: Base dependencies are `fastapi, uvicorn, gunicorn, python-dotenv, httpx` plus additional packages for file and image parsing: `openpyxl, python-docx, python-pptx, pdfplumber, pillow`.

### How It Works
- The bot listens to the Lark webhook for events.
- For each event:
  - `text` messages are forwarded directly to OpenAI for chat completion.
  - `image` messages are downloaded, encoded to Base64 and wrapped in a multi‑modal payload for OpenAI to process.
  - `file` messages are downloaded and parsed (Excel, Word, PPT, PDF) using appropriate Python libraries, then sent to OpenAI for summarization or question answering.
- Responses are sent back to the user in Lark.

### API
- `POST /webhook` – Handles Lark events (text, image, file), routes them to the correct processing logic and returns an OpenAI response.
- `GET /health` – Health check endpoint.

### FAQ
**Can it handle images/files?** Yes! The bot supports images and files like Excel, Word, PowerPoint and PDF. The contents are parsed and used to generate a summary or answer.
