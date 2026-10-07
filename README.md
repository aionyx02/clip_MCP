# clip-mcp

以時間軸為核心、底層為 FFmpeg 的影片剪輯 MCP 伺服器，附本機瀏覽器編輯器。
分析、轉錄、說話者切分、人臉偵測與語意檢索全部在本機執行；素材檔案不會離開這台電腦。

## 設計原則

- **判斷交給模型，算術交給伺服器。** 模型只決定選哪些片段與理由；剪點位置、呼吸長度、段落合併、成片長度
  都由確定性的純函式計算。同一份計畫必定編出同一條時間軸。
- **計畫是來源，時間軸是產物。** 模型寫宣告式 plan（目標、段落、選材與理由、排除理由），由編譯器產生時間軸。
  手動改過的片段視為鎖定，重編時保留；新計畫若會丟掉鎖定片段則拒絕編譯。
- **驗證失敗即退回，不默默修正。** 錯誤附理由回傳；做不到的功能明講，不以近似結果冒充。
- **量測，不評價。** 分析輸出數值（曝光、模糊、抖動、響度、人臉位置），判斷留給呼叫端。

## 架構

```
MCP client ──stdio──> server.py ──> engine/ ──> FFmpeg
                         │            (不依賴 MCP，可單獨使用)
editor (ui/) ──HTTP──────┤
                         └──> storage/  SQLite（現況） + git 歷史（dulwich，所有版本）
```

| 模組 | 職責 |
|---|---|
| `server.py` | MCP 工具介面與參數驗證 |
| `engine/` | 分析、語意時間軸、計畫編譯、filtergraph 建構、render、字幕、交換格式 |
| `models/` | Pydantic 領域模型（素材、語意片段、計畫、時間軸、工作） |
| `storage/` | SQLite 持久化、版本歷史、儲存空間管理 |
| `ui/` | 編輯器：與 AI 共用同一組工具與驗證規則 |
| `benchmark/` | 剪輯結果的確定性評分 |

詳見 [docs/architecture.md](docs/architecture.md)。

## 功能概要

| 領域 | 內容 |
|---|---|
| 分析 | 單次解碼完成換場、黑畫面、凍結、靜音、逐鏡頭畫質、逐秒音訊品質與人臉；faster-whisper 詞級逐字稿（zh-TW / zh-HK / zh-Hans）；ONNX 說話者切分，跨檔案一致標籤；分析帶配方，設定變更即標為過期 |
| 語意時間軸 | 句子／鏡頭／停頓為單位的可檢索片段，各帶實測安全剪點；條件與語意檢索；由實測訊號產生候選段落邊界，模型只能在候選內裁決 |
| 剪輯 | 磁性時間軸、J/L cut、轉場、變速（保留音高）、B-roll、子母畫面、逐片段調色；轉場與 J/L cut 不改變成片長度；修剪吸附詞邊界 |
| 計畫 | 宣告式 plan、單項修改、整體修改（節奏、配樂音量、整段移除）、版本比較與視覺化差異、回退 |
| 聲音 | 多軌配樂、人聲 ducking、隨段落換曲、剪點對拍、語音修復、逐講者響度、輸出統一 -14 LUFS |
| 字幕 | 綁定素材時間而非成片時間，隨剪輯移動；依詞斷行、平台安全區樣式、逐字亮、雙語、講者標色；ASS 燒錄 / SRT |
| 輸出 | render 前檢查（黑畫面、爆音、字幕出界、長度）；橫／直／方形三比例，跟臉裁切；章節、封面；FCPXML / OTIO / EDL 匯出並列出無法攜帶的屬性 |
| 版本 | 專案、計畫、字幕的每次變更都是 git commit，可讀、可回退、可分支、可標記 |
| 執行 | render 與分析在獨立 worker 執行，依工作數與可用記憶體排隊，可查進度、可取消、跨重啟持續 |

目前不支援：靜態圖片、字幕以外的文字圖形、調色以外的濾鏡、同軌畫面重疊。

## 資料邊界

| 留在本機 | 回傳給 MCP 客戶端 |
|---|---|
| 影音原檔、預覽、成品 | 檔名、路徑、規格 |
| 逐字稿、量測、向量索引、模型權重 | 查詢命中的片段文字與量測值 |
| 專案、計畫、版本歷史 | 模型要求時的縮圖拼貼與音量圖 |

- 沒有工具回傳影音本體。模型仍可主動讀取完整逐字稿（`get_analysis`、`query_clips` 的 `brief`、`get_subtitles`）。
- 回傳內容的去向由客戶端決定；搭配本機模型（如 LM Studio）則完全離線。
- 對外連線僅限：模型權重下載（固定版本、驗證 SHA-256）、安裝程式下載 FFmpeg、安裝版的更新檢查。

## 安裝

需求：Python 3.14+、[uv](https://docs.astral.sh/uv/)、FFmpeg / ffprobe 7.0+ 位於 `PATH`。
Windows 另有安裝程式（`installer/`），自帶 Python 與 FFmpeg。

```bash
git clone https://github.com/aionyx02/clip_MCP.git
cd clip_MCP
uv tool install --editable .
clip-mcp setup        # 選擇工作區，逐一確認要註冊的客戶端（--dry-run 預覽、--all、--client <name>）
clip-mcp check        # 檢查工作區、FFmpeg 與各客戶端註冊狀態
clip-mcp ui           # 開啟編輯器
```

支援自動註冊：Claude Code、Claude Desktop、Codex（含 ChatGPT 桌面版 Codex 分頁）、opencode、Gemini CLI、
LM Studio、Cherry Studio（一鍵安裝連結）。修改前備份為 `<檔名>.clip-mcp.bak`；重複執行為冪等。
其他 stdio 客戶端將 command 設為 `clip-mcp`（完整路徑見 `clip-mcp check`），不需參數。

不要以 `uv run clip-mcp` 註冊：每次啟動會重寫 `.venv` 內的執行檔，多客戶端同時使用時會因檔案鎖定失敗。

移除：`clip-mcp uninstall`（只移除自己加入的設定；工作區先詢問，成品不動），再 `uv tool uninstall clip-mcp`。

除錯：`uv run fastmcp dev fastmcp.json` 開啟 MCP Inspector。

## 工具

使用指南以 MCP resource `skill://clip-editing/SKILL.md` 提供（不支援 resource 的客戶端用 `read_resource`）。

| 類別 | 工具 |
|---|---|
| 素材庫 | `inspect_media` `import_asset` `import_folder` `list_assets` `edit_asset` `organize_library` `move_files` |
| 分析 | `analyze_asset` `listen_again` `get_analysis` `view_frames` |
| 語意時間軸 | `build_semantic_timeline` `query_clips` `get_semantic_clip` `propose_sections` `set_sections` `frames_for_clips` `set_clip_tags` |
| 計畫 | `save_plan` `amend_plan` `copy_plan` `get_plan` `validate_plan` `compile_plan` `diff_plan` `preview_plan_diff` `list_plan_versions` `revert_plan` `propose_broll` |
| 專案 | `create_project` `list_projects` `get_project` `apply_edits` `preview_project` `preview_sound` |
| 字幕 | `generate_subtitles` `get_subtitles` |
| 輸出 | `check_render` `render_project` `view_render` `get_chapters` `propose_covers` `export_cover` `export_timeline` |
| 版本 | `project_history` `restore_version` `branch_project` `mark_version` |
| 工作與儲存 | `get_job` `cancel_job` `storage_usage` `clean_storage` `tidy_old_outputs` |

## 工作區

解析順序：`CLIP_MCP_WORKSPACE` → `clip-mcp setup` 寫入的指標檔 → 原始碼安裝用 `./workspace`，
套件安裝用使用者資料夾。模型權重預設存放於工作區內（`models/`），刪除工作區即一併移除。

| 變數 | 預設 | 說明 |
|---|---|---|
| `CLIP_MCP_WORKSPACE` | 見上 | 工作區路徑 |
| `CLIP_MCP_MODELS` | `<workspace>/models` | 模型權重位置，可指向共用目錄 |
| `CLIP_MCP_WHISPER_MODEL` | `large-v3-turbo`(自動偵測電腦性能並選擇) | 語音辨識模型 |
| `CLIP_MCP_WHISPER_DEVICE` | `auto` | `cuda` / `cpu` |
| `CLIP_MCP_SUBTITLE_FONT` | `Arial` | 燒錄字幕字型 |
| `CLIP_MCP_MAX_JOBS` | `2` | 同時執行的工作上限 |
| `CLIP_MCP_MEMORY_RESERVE_MB` | `2048` | 保留給系統的實體記憶體 |
| `CLIP_MCP_MAX_DECODERS` | `4` | 抽幀解碼行程上限 |

## 測試

```bash
uv run pytest
```

測試以 FFmpeg 即時產生素材並使用暫存工作區；部分測試實際 render 後量測（響度、ducking、轉場、同步、字幕位置、跟臉裁切、節拍）。

`src/app/benchmark/` 對編譯結果做確定性評分（切字剪點、壞幀剪點、長度誤差、必留覆蓋率、必剔洩漏率、剪點餘裕）。
語料庫（`corpus/`）尚在建立，目前尚無基準分數；進度見 [docs/roadmap.md](docs/roadmap.md)。

## 授權

[GNU AGPL v3](LICENSE)
