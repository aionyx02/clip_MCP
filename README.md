# clip-mcp

以時間軸為核心的影片剪輯 MCP 伺服器，底層是 FFmpeg。

把「把這三支接起來」「剪掉講錯的地方」「加背景音樂、配字幕」這類說法，
交給支援 MCP 的 AI 客戶端，由它呼叫本伺服器的工具完成剪輯並輸出 MP4。
分析、轉檔、語音辨識全部在本機跑，**素材檔案本身不會被上傳到任何地方**。

要注意的是：`view_frames`、`preview_project`、`frames_for_clips` 會把縮圖當成
工具結果回傳給你的 MCP 客戶端，逐字稿與查詢結果也一樣是文字回傳。如果那個客戶端是
雲端模型，這些縮圖與文字就會離開本機。伺服器不主動上傳任何東西，但它也管不到
客戶端拿到結果之後送去哪裡——這件事取決於你接的是哪個客戶端。

## 設計上的一句話

**判斷交給模型與使用者，算術交給伺服器。** 模型決定選哪些片段、為什麼；
從那裡開始的每一件事——剪點落在哪、留多少呼吸、哪些併成一段、成片多長——
都是確定性的純函式。所以同一份計畫必定編出同一個結果，
而模型最糟的錯誤是「段落分得不好」，不會是「切到字中間」。

驗證不過的東西原樣退回並附上理由，不默默修正。

## 它能做什麼

- **看懂素材**：換場、黑畫面、靜止畫面、靜音偵測，加上詞級時間戳的本機逐字稿
  （faster-whisper，可轉成 zh-TW / zh-HK / zh-Hans）。
- **看見畫面**：把抽出的影格拼成一張標了時間的縮圖總覽（`view_frames`，一次可以吃多支
  素材），或把剪好的序列拼成 storyboard（`preview_project`），幾秒就能確認再決定要不要 render。
- **語意時間軸**：把分析結果拆成可檢索的語意片段——一句話、一顆鏡頭、一段停頓，
  每段都帶著自己實測的安全剪點（`build_semantic_timeline`、`query_clips`）。
  再往上由客戶端模型把片段裁決成段落與主題（`propose_sections`、`set_sections`）——
  伺服器用實測訊號**過量產生**候選邊界，模型只能在候選裡挑選與命名，
  所以段落永遠長在量出來的靜音邊緣上。看圖寫回的描述會標明出處
  （`frames_for_clips`、`set_clip_tags`）。
- **剪輯計畫**：先寫一份宣告式的 plan——目標、段落、選了哪些片段、為什麼、
  哪些看過沒用——再由伺服器編譯成時間軸（`save_plan`、`validate_plan`、`compile_plan`）。
  同一份 plan 必定編出同一個結果，要改某一項用 `amend_plan` 就好，不必整份重送，
  兩種剪法可以用 `diff_plan` 直接比較。
- **出身與鎖定**：編譯出來的每個片段都記得它從哪份計畫、哪些語意片段來。
  **手動改過就等於鎖定**，重新編譯時你的修改原樣保留，只有順序由新計畫決定；
  新計畫如果不再用到某個鎖定片段的素材，整個編譯會被拒絕並列出擋路的片段——
  丟掉使用者的工作比停下來糟。
- **磁性剪輯**：插入、修剪、刪除、搬移、切開、重排，後面的片段會自己讓位或補上空隙，
  不用自己算秒數。支援倒敍這類非線性順序。
- **剪點不落在字中間**：`head` / `tail` 這種會落在任意秒數的修剪會吸附到最近的詞邊界，
  「呼吸」把剪點推進詞裡時會把呼吸還回去。修不掉的會回報，不會默默留著。
  句子沒講完就切掉的地方也會回報還差幾秒——那會改變成片長度，是人要決定的事。
- **聲音**：多條音軌背景音樂、逐片段音量與淡入淡出、`duck_under_speech`
  讓音樂在人聲出現時自動退下、`fit_track` 把音樂對齊影片長度（裁切、接續、循環、搬移淡出）。
- **統一響度**：每次輸出都正規化到 -14 LUFS，不同裝置錄的素材不會一段大聲一段小聲。
- **畫面處理**：逐片段亮度／對比／飽和度／色溫（含黑白）、淡入淡出黑、
  畫中畫（以畫面比例定位的子母畫面或跳接畫面）。
- **字幕**：從逐字稿產生字幕，**綁在素材的秒數上而不是成片的秒數上**——
  剪輯搬動時字幕自己跟著走，刪掉的片段會把它的字幕一起帶走，
  從一句話中間切開則兩邊各出現一次。依詞斷行、避開手機直式影片的操作區，
  輸出 ASS 並可燒錄進畫面。改錯字只要改那一句。
- **背景工作**：render 與分析在獨立 worker 行程執行，有進度可查、可取消，伺服器重啟也不會中斷。
- **持久化**：專案、素材、分析結果、計畫、工作狀態都存在 SQLite，隔幾天回來還能接著剪。
- **內建使用指南**：`clip-editing` skill 以 MCP resource 提供，不綁特定模型，
  教 AI 如何把需求翻成工具呼叫、什麼時候該先問、預設值怎麼挑。

目前還不支援：靜態圖片、變速、疊化與轉場特效（只有淡入淡出黑）、
J/L cut、字幕以外的文字與圖形，以及上述以外的濾鏡。
需要這些時伺服器會明講做不到，並提出最接近的做法，而不是假裝做了。

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

**素材與分析**

| 工具 | 用途 |
|---|---|
| `inspect_media` | 讀取檔案的解析度、長度、幀率等技術資訊 |
| `import_asset` / `import_folder` | 把檔案或整個資料夾註冊成素材（可遞迴、自然排序） |
| `list_assets` | 列出已匯入的素材 |
| `analyze_asset` | 背景分析：換場、黑畫面、靜止、靜音、逐字稿 |
| `get_analysis` | 依時間區間讀取分析結果 |
| `view_frames` | 抽幀拼成標時間的縮圖總覽，一次可吃多支素材 |

**語意時間軸**

| 工具 | 用途 |
|---|---|
| `build_semantic_timeline` | 由分析結果推導可檢索的語意片段（同樣輸入必得同一組 ID） |
| `query_clips` | 依種類、文字、標籤、長度、分數等條件檢索片段 |
| `get_semantic_clip` | 讀單一片段的完整文字，可含每個詞的時間 |
| `propose_sections` / `set_sections` | 產生候選邊界，由模型在候選裡裁決並命名段落與主題 |
| `frames_for_clips` / `set_clip_tags` | 看圖並把描述寫回片段，標明出處 |

**剪輯計畫**

| 工具 | 用途 |
|---|---|
| `save_plan` | 存下一份計畫：目標、段落、選材與理由、被排除的理由 |
| `amend_plan` | 只改其中一項（重設修剪、改寫理由、加一段、拿掉一段） |
| `get_plan` / `diff_plan` | 讀回計畫，或比較兩份計畫實際差在哪 |
| `validate_plan` | 編譯前檢查，附上問題與提醒 |
| `compile_plan` | 把計畫編譯成時間軸 |

**專案與輸出**

| 工具 | 用途 |
|---|---|
| `create_project` / `list_projects` / `get_project` | 建立與讀取專案（名稱、輸出尺寸、幀率、軌道、片段、版本） |
| `apply_edits` | 以樂觀鎖批次套用編輯操作 |
| `preview_project` | 剪輯後序列的 storyboard |
| `generate_subtitles` / `get_subtitles` | 產生與讀取字幕 |
| `render_project` | 背景輸出 MP4（可選 480p 快速預覽、燒字幕、響度目標） |
| `get_job` / `cancel_job` | 查詢進度與取消背景工作 |

`apply_edits` 接受的操作：`add_track`、`add_clip`、`insert_clip`、`trim_clip`、
`delete_clip`、`move_clip`、`split_clip`、`reorder_clip`、`fit_track`、
`set_clip_look`、`set_clip_audio`、`set_track_audio`、`set_clip_pinned`、
`set_subtitles`、`edit_subtitle`、`rename_project`。

## 典型流程

1. `import_folder` 匯入素材，`view_frames` 一次看幾支，快速把整批掃過。
2. 需要依內容剪輯時，`analyze_asset` 分析（不需要逐字稿就關掉 `transcribe`，快很多）。
3. `build_semantic_timeline` 之後用 `query_clips` 找素材，不要從頭讀逐字稿。
4. 素材多的時候先 `propose_sections` / `set_sections` 分段，再從段落裡挑句子。
5. `save_plan` 寫下要剪什麼、為什麼，`validate_plan` 檢查，`compile_plan` 編譯成時間軸。
6. `preview_project` 看 storyboard 確認剪點。
7. `generate_subtitles` 檢查文字後用 `set_subtitles` 存起來。
8. `render_project` 先出 `is_preview: true` 預覽，滿意後再輸出完整品質。

改主意時改計畫再重編，不要直接推時間軸上的片段——理由記在計畫裡，
時間軸只是計畫掉出來的結果。手改過的片段重編時不會被洗掉。

短一點的剪輯可以跳過計畫，直接 `create_project` 加軌道、用 `insert_clip` 排序列。

## 工作區與環境變數

素材資料庫、暫存與成品都放在 `workspace/`：

```
workspace/
├── clip_mcp.db        # 專案、素材、分析、計畫、工作狀態（SQLite）
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

## 測試與評測

```bash
uv run pytest
```

測試會用 FFmpeg 即時產生素材，並在暫存目錄開獨立工作區執行，不會動到 `workspace/`。
部分測試會實際 render 後量測響度、調色與字幕位置，所以需要 FFmpeg 在 `PATH` 上。

`src/app/benchmark/` 是另一層：不 render，直接替一份編譯好的剪輯打分——
切到字的剪點、落在壞幀上的剪點、成片長度誤差、必留段落覆蓋率、必剔段落洩漏率，
以及每個剪點還有多少餘裕。全是純函式，所以同一份計畫對同一批素材永遠得到同一個分數。
量不到的東西（例如沒分析過的素材）會被點名，不會混進「沒問題」裡。
語料庫與更上層的成片評分還沒做，見 [docs/roadmap.md](docs/roadmap.md) §11。

## 架構

各模組職責見 [docs/architecture.md](docs/architecture.md)。
引擎層（`src/app/engine/`）不依賴 MCP，可以單獨使用。
2.0 的方向與里程碑見 [docs/roadmap.md](docs/roadmap.md)。

## 授權

[GNU AGPL v3](LICENSE)
