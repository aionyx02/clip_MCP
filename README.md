# clip-mcp

以時間軸為核心的影片剪輯 MCP 伺服器，底層是 FFmpeg。

把「把這三支接起來」「剪掉講錯的地方」「加背景音樂、配字幕」這類說法，
交給支援 MCP 的 AI 客戶端，由它呼叫本伺服器的工具完成剪輯並輸出 MP4。
分析、轉檔、語音辨識全部在本機跑，素材不會上傳到任何地方。

## 它能做什麼

- **看懂素材**：換場、黑畫面、靜止畫面、靜音偵測，加上詞級時間戳的本機逐字稿
  （faster-whisper，可轉成 zh-TW / zh-HK / zh-Hans）。
- **看見畫面**：把抽出的影格拼成一張標了時間的縮圖總覽（`view_frames`），
  或把剪好的序列拼成 storyboard（`preview_project`），幾秒就能確認再決定要不要 render。
- **磁性剪輯**：插入、修剪、刪除、搬移、切開、重排，後面的片段會自己讓位或補上空隙，
  不用自己算秒數。支援倒敍這類非線性順序。
- **聲音**：多條音軌背景音樂、逐片段音量與淡入淡出、`duck_under_speech`
  讓音樂在人聲出現時自動退下、`fit_track` 把音樂對齊影片長度（裁切、接續、循環、搬移淡出）。
- **統一響度**：每次輸出都正規化到 -14 LUFS，不同裝置錄的素材不會一段大聲一段小聲。
- **畫面處理**：逐片段亮度／對比／飽和度／色溫（含黑白）、淡入淡出黑、
  畫中畫（以畫面比例定位的子母畫面或跳接畫面）。
- **字幕**：從逐字稿產生對齊剪輯後時間軸的字幕，依詞斷行、避開手機直式影片的操作區，
  輸出 ASS 並可燒錄進畫面。
- **背景工作**：render 與分析在獨立 worker 行程執行，有進度可查、可取消，伺服器重啟也不會中斷。
- **持久化**：專案、素材、分析結果、工作狀態都存在 SQLite，隔幾天回來還能接著剪。
- **內建使用指南**：`clip-editing` skill 以 MCP resource 提供，不綁特定模型，
  教 AI 如何把需求翻成工具呼叫、什麼時候該先問、預設值怎麼挑。

目前還不支援：靜態圖片、變速、疊化與轉場特效（只有淡入淡出黑）、
J/L cut、字幕以外的文字與圖形，以及上述以外的濾鏡。

接下來要往哪走見 [docs/roadmap.md](docs/roadmap.md)。

## 需求

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- FFmpeg 與 ffprobe，且在 `PATH` 上

## 安裝

```bash
git clone https://github.com/aionyx02/clip_MCP.git
cd clip_MCP
uv sync
```

第一次執行語音辨識時會下載 `large-v3-turbo` 模型。

## 接到 AI 客戶端

專案內已附 `.mcp.json`，在此目錄啟動 Claude Code 即可直接使用。

其他客戶端（Claude Desktop 等）加入這段設定：

```json
{
  "mcpServers": {
    "clip-mcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/clip_MCP", "run", "clip-mcp"]
    }
  }
}
```

伺服器走 stdio transport。也可以用 `uv run fastmcp run fastmcp.json` 啟動，
或 `uv run fastmcp dev fastmcp.json` 開 MCP Inspector 除錯。

## 工具

| 工具 | 用途 |
|---|---|
| `inspect_media` | 讀取檔案的解析度、長度、幀率等技術資訊 |
| `import_asset` / `import_folder` | 把檔案或整個資料夾註冊成素材（可遞迴、自然排序） |
| `list_assets` | 列出已匯入的素材 |
| `analyze_asset` | 背景分析：換場、黑畫面、靜止、靜音、逐字稿 |
| `get_analysis` | 依時間區間讀取分析結果 |
| `view_frames` | 抽幀拼成標時間的縮圖總覽 |
| `create_project` / `list_projects` / `get_project` | 建立與讀取專案（輸出尺寸、幀率、軌道、片段、版本） |
| `apply_edits` | 以樂觀鎖批次套用編輯操作 |
| `preview_project` | 剪輯後序列的 storyboard |
| `generate_subtitles` / `get_subtitles` | 產生與讀取字幕 |
| `render_project` | 背景輸出 MP4（可選 480p 快速預覽、燒字幕、響度目標） |
| `get_job` / `cancel_job` | 查詢進度與取消背景工作 |

`apply_edits` 接受的操作：`add_track`、`add_clip`、`insert_clip`、`trim_clip`、
`delete_clip`、`move_clip`、`split_clip`、`reorder_clip`、`fit_track`、
`set_clip_look`、`set_clip_audio`、`set_track_audio`、`set_subtitles`、`edit_subtitle`。

## 典型流程

1. `import_folder` 匯入素材，`view_frames` 快速看過每支影片。
2. 需要依內容剪輯時，`analyze_asset` 分析（不需要逐字稿就關掉 `transcribe`，快很多）。
3. `create_project` 決定輸出尺寸，`apply_edits` 加軌道、用 `insert_clip` 排序列。
4. 加音樂的音軌開 `duck_under_speech`，最後用 `fit_track` 對齊長度。
5. `preview_project` 看 storyboard 確認剪點。
6. `generate_subtitles` 檢查文字後用 `set_subtitles` 存起來。
7. `render_project` 先出 `is_preview: true` 預覽，滿意後再輸出完整品質。

## 工作區與環境變數

素材資料庫、暫存與成品都放在 `workspace/`：

```
workspace/
├── clip_mcp.db        # 專案、素材、分析、工作狀態（SQLite）
├── outputs/<job_id>/  # 每次 render 各自一個目錄，不會互相覆蓋
└── jobs/<job_id>/     # 分析工作的暫存
```

| 變數 | 預設 | 說明 |
|---|---|---|
| `CLIP_MCP_WORKSPACE` | `workspace` | 工作區路徑 |
| `CLIP_MCP_WHISPER_MODEL` | `large-v3-turbo` | 語音辨識模型 |
| `CLIP_MCP_WHISPER_DEVICE` | `auto` | `cuda` 或 `cpu`；auto 會偵測 CUDA，失敗時退回 CPU |
| `CLIP_MCP_SUBTITLE_FONT` | `Arial` | 燒錄字幕的字型 |
| `CLIP_MCP_MAX_JOBS` | `2` | 同時執行的 render／分析工作數上限 |
| `CLIP_MCP_MEMORY_RESERVE_MB` | `2048` | 保留給系統、工作不得動用的實體記憶體 |
| `CLIP_MCP_MAX_DECODERS` | `4` | 縮圖抽幀同時開啟的解碼行程上限（整個伺服器共用）|

### 記憶體與排隊

render 與分析都是重工作：一次 render 會為每個片段各開一個解碼器，一次轉錄要載入數 GB
的語音模型。所以工作不是「呼叫就跑」，而是先排隊——只有在工作數與剩餘實體記憶體
都夠時才真正啟動 worker。排隊中的工作只佔一筆資料庫紀錄，`get_job` 的 `stage`
會說明它在等什麼；等待是正常的，照常輪詢即可。

機器記憶體不足時，**唯一**一個工作仍然會啟動：全部拒絕比慢慢跑更糟。真正跑不動的情況
（例如 CPU 轉錄但可用記憶體低於模型需求）會直接失敗並告訴你改用哪個較小的模型，
而不是把整台電腦拖垮。

## 測試

```bash
uv run pytest
```

測試會用 FFmpeg 即時產生素材，並在暫存目錄開獨立工作區執行，不會動到 `workspace/`。
部分測試會實際 render 後量測響度、調色與字幕位置，所以需要 FFmpeg 在 `PATH` 上。

## 架構

各模組職責見 [docs/architecture.md](docs/architecture.md)。
引擎層（`src/app/engine/`）不依賴 MCP，可以單獨使用。
2.0 的方向與里程碑見 [docs/roadmap.md](docs/roadmap.md)。

## 授權

[GNU AGPL v3](LICENSE)
