# clip-mcp

以時間軸為核心的影片剪輯 MCP 伺服器，底層是 FFmpeg。

把「把這三支接起來」「剪掉講錯的地方」「加背景音樂、配字幕」這類說法，
交給支援 MCP 的 AI 客戶端，由它呼叫本伺服器的工具完成剪輯並輸出 MP4。
分析、轉檔、語音辨識、說話者切分、人臉偵測全部在本機跑，**素材檔案本身不會被上傳到任何地方**。
唯一的對外連線是第一次用到某個模型時去抓權重，抓完就留在工作區裡，之後完全離線。

要注意的是：`view_frames`、`preview_project`、`frames_for_clips`、`propose_covers` 會把縮圖當成
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
- **量出拍得好不好**：同一次解碼順便逐鏡頭量曝光、對比、模糊、動態與鏡頭抖動，
  逐秒量響度、峰值、底噪與削波，並逐秒偵測畫面裡有幾張臉、最大的那張佔多少、在哪。
  這些是量測不是評價——伺服器不告訴你哪顆算太暗，它給你數字，
  讓 `query_clips` 可以直接下「她在講話、鏡頭是穩的、而且人在畫面左邊」。
  分析帶著**配方**（版本、偵測參數、用了哪些模型），換過設定的素材會被標成過期。
- **誰在講話**：對談與訪談會自動切分說話者，而且**跨檔案認人**——兩台機拍同一場訪談，
  同一個人在兩支檔案裡都是 `V1`，片段與字幕用的是同一組標籤。
  跨到中間換人的片段**不給標籤**——寧可空著，也不要給一個會被燒進字幕的錯名字。
  知道有幾個人就告訴它（`speakers`），它就只會找出那麼多個。
- **看見畫面**：把抽出的影格拼成一張標了時間的縮圖總覽（`view_frames`，一次可以吃多支
  素材），或把剪好的序列拼成 storyboard（`preview_project`），幾秒就能確認再決定要不要 render。
- **語意時間軸**：把分析結果拆成可檢索的語意片段——一句話、一顆鏡頭、一段停頓，
  每段都帶著自己實測的安全剪點（`build_semantic_timeline`、`query_clips`）。
  再往上由客戶端模型把片段裁決成段落與主題（`propose_sections`、`set_sections`）——
  伺服器用實測訊號**過量產生**候選邊界，模型只能在候選裡挑選與命名，
  所以段落永遠長在量出來的靜音邊緣上。看圖寫回的描述會標明出處
  （`frames_for_clips`、`set_clip_tags`）。
- **剪輯的基本盤**：J / L cut（聲音先進或後留）、三種轉場（溶接、擦劃、過一個顏色）、
  變速（保留音高或跟著變調）、段落標記。J / L cut 與轉場都**不移動任何 clip**——
  剪點留在原地，多出來的媒體從片段的來源範圍之外取，所以成片長度不變。
  段落標記由 `compile_plan` 從 plan 的 beats 寫上去，手加的那些不會被下一次編譯洗掉。
- **剪輯計畫**：先寫一份宣告式的 plan——目標、段落、選了哪些片段、為什麼、
  哪些看過沒用——再由伺服器編譯成時間軸（`save_plan`、`validate_plan`、`compile_plan`）。
  同一份 plan 必定編出同一個結果，要改某一項用 `amend_plan` 就好，不必整份重送，
  兩種剪法可以用 `diff_plan` 直接比較。編譯時順便做完粗剪的清理：拿掉氣口、
  同一句講好幾次只留最後講完的那次、頭尾修乾淨、剪點避開黑畫面與凍結畫面。
- **回饋與版本**：每一版 plan 都留著，`revert_plan` 回到任何一版（回退本身也是新的一版，
  什麼都不會丟）。「整體節奏太慢」「音樂太大聲」「這整段不要」各有一個一次改整支的修改，
  不必逐段改。`preview_plan_diff` 把這一輪改了什麼畫在一張 storyboard 上。
- **B-roll**：在剪好的序列上蓋畫面、底下的聲音繼續跑。`propose_broll` 指出畫面停太久的地方，
  覆蓋規則（每段多長、一段最多蓋幾成、不能蓋掉標為關鍵的鏡頭）寫在編譯器裡，違規就退回。
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
  plan 裡的配樂可以**隨段落換曲**，也可以要求**剪點對到拍上**（節拍在分析時量出來，
  剪點只在靜音裡移動，不會切到字）。語音修復（低頻轟聲、嘶聲、齒音，一律明確開啟）
  與**逐講者響度一致**（兩個人錄得一大一小時各自調）。
- **統一響度**：每次輸出都正規化到 -14 LUFS，不同裝置錄的素材不會一段大聲一段小聲。
- **畫面處理**：逐片段亮度／對比／飽和度／色溫（含黑白）、淡入淡出黑、
  畫中畫（以畫面比例定位的子母畫面或跳接畫面）。
- **一支剪輯出三種比例**：同一個專案 render 成橫式、直式、方形，形狀不合的鏡頭**跟著臉裁**
  而不是硬裁中間——臉移出去時用切的換構圖，不推鏡。
- **字幕**：從逐字稿產生字幕，**綁在素材的秒數上而不是成片的秒數上**——
  剪輯搬動時字幕自己跟著走，刪掉的片段會把它的字幕一起帶走，
  從一句話中間切開則兩邊各出現一次。依詞斷行，YouTube／Reels／TikTok 各有一組
  避開平台按鈕的樣式，可以一個字一個字亮、雙語兩行、標出誰在講話並換色。
  輸出 ASS 並可燒錄進畫面，也能匯出 SRT。改錯字只要改那一句。
- **預覽**：storyboard 幾秒就出來，而且裁切方式跟 render 一樣。預覽 render 會記住每顆鏡頭的畫面，
  改一個剪點只重算那一顆。`preview_sound` 只算聲音，給一張人聲對配樂的圖和一個可以聽的檔案——
  AI 聽不到，這是它確認音樂有沒有蓋過講話的方法。
- **出片**：render 前先檢查黑畫面、爆音、字幕出界、長度不符，有問題就擋下來等使用者決定；
  章節從段落長出來（也寫進 MP4），封面候選一顆鏡頭一張；剪輯可以匯出成
  FCPXML／OTIO／EDL，給 Premiere、DaVinci Resolve、Final Cut 接手細修。
- **背景工作**：render 與分析在獨立 worker 行程執行，有進度可查、可取消，伺服器重啟也不會中斷。
- **持久化**：專案、素材、分析結果、計畫、工作狀態都存在 SQLite，隔幾天回來還能接著剪。
- **內建使用指南**：`clip-editing` skill 以 MCP resource 提供，不綁特定模型，
  教 AI 如何把需求翻成工具呼叫、什麼時候該先問、預設值怎麼挑。

目前還不支援：靜態圖片、字幕以外的文字與圖形，以及調色以外的濾鏡。
需要這些時伺服器會明講做不到，並提出最接近的做法，而不是假裝做了。

功能都做齊了，**還沒做的是證明剪得好**：下一步是用真實素材做 benchmark。
這件事、以及刻意不做的事為什麼不做，見 [docs/roadmap.md](docs/roadmap.md)。

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
| `list_assets` | 列出已匯入的素材，並標明哪些分析過、哪些已過期 |
| `analyze_asset` | 背景分析：一次解碼跑完換場、黑畫面、靜止、靜音、逐鏡頭畫質、逐秒音訊品質與人臉，之後接逐字稿與說話者切分 |
| `get_analysis` | 依時間區間讀取分析結果，含逐鏡頭量測、該區間的音訊與人臉摘要、說話者段落 |
| `view_frames` | 抽幀拼成標時間的縮圖總覽，一次可吃多支素材 |
| `list_resources` / `read_resource` | 讀內建的使用指南（給不支援 MCP resource 的客戶端） |

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
| `save_plan` | 存下一份計畫：目標、段落、選材與理由、被排除的理由、配樂、B-roll、節奏 |
| `amend_plan` | 只改其中一項，或一次改整支（節奏、配樂音量、目標、整段拿掉） |
| `get_plan` / `diff_plan` | 讀回計畫（任何一版），或比較兩份計畫、兩個版本實際差在哪 |
| `list_plan_versions` / `revert_plan` | 列出每一版改了什麼，回到其中一版 |
| `preview_plan_diff` | 把兩個版本的差異畫在一張 storyboard 上 |
| `validate_plan` | 編譯前檢查，附上問題與提醒 |
| `propose_broll` | 指出畫面停太久、值得蓋 B-roll 的地方 |
| `compile_plan` | 把計畫編譯成時間軸 |

**專案與輸出**

| 工具 | 用途 |
|---|---|
| `create_project` / `list_projects` / `get_project` | 建立與讀取專案（名稱、輸出尺寸、幀率、軌道、片段、版本） |
| `apply_edits` | 以樂觀鎖批次套用編輯操作 |
| `preview_project` | 剪輯後序列的 storyboard（可選其他比例） |
| `preview_sound` | 只算聲音：人聲對配樂的圖與可以聽的混音檔 |
| `generate_subtitles` / `get_subtitles` | 產生與讀取字幕 |
| `check_render` | render 前檢查黑畫面、爆音、字幕出界、長度 |
| `render_project` | 背景輸出 MP4（預覽只重算改過的鏡頭；可選比例、燒字幕、響度目標） |
| `get_chapters` | 從段落產生 YouTube 章節 |
| `propose_covers` / `export_cover` | 封面候選與全尺寸封面 |
| `export_timeline` | 匯出 FCPXML／OTIO／EDL，或把字幕匯出成 SRT |
| `get_job` / `cancel_job` | 查詢進度與取消背景工作 |

`apply_edits` 接受的操作：`add_track`、`add_clip`、`insert_clip`、`trim_clip`、
`delete_clip`、`move_clip`、`split_clip`、`reorder_clip`、`fit_track`、
`set_clip_look`、`set_clip_audio`、`set_clip_speed`、`set_track_audio`、`set_clip_pinned`、
`set_markers`、`set_subtitles`、`edit_subtitle`、`set_caption_style`、`rename_project`。

## 典型流程

1. `import_folder` 匯入素材，`view_frames` 一次看幾支，快速把整批掃過。
2. 需要依內容剪輯時，`analyze_asset` 分析（不需要逐字稿就關掉 `transcribe`，快很多）。
3. `build_semantic_timeline` 之後用 `query_clips` 找素材，不要從頭讀逐字稿。
4. 素材多的時候先 `propose_sections` / `set_sections` 分段，再從段落裡挑句子。
5. `save_plan` 寫下要剪什麼、為什麼，`validate_plan` 檢查，`compile_plan` 編譯成時間軸。
6. `preview_project` 看 storyboard 確認剪點，有配樂就再用 `preview_sound` 確認聲音。
7. `generate_subtitles` 檢查文字後用 `set_subtitles` 存起來。
8. `check_render` 看有沒有要先處理的，`render_project` 先出 `is_preview: true` 預覽，
   滿意後再輸出完整品質；要發到多個平台就每個比例各 render 一次。

改主意時改計畫再重編，不要直接推時間軸上的片段——理由記在計畫裡，
時間軸只是計畫掉出來的結果。手改過的片段重編時不會被洗掉。

短一點的剪輯可以跳過計畫，直接 `create_project` 加軌道、用 `insert_clip` 排序列。

## 工作區與環境變數

素材資料庫、暫存與成品都放在 `workspace/`：

```
workspace/
├── clip_mcp.db        # 專案、素材、分析、計畫、工作狀態（SQLite）
├── models/            # 所有模型權重，用到才下載
│   ├── whisper/       #   語音辨識（large-v3-turbo 約 1.5 GB）
│   ├── diarization/   #   說話者切分（約 33 MB）
│   └── faces/         #   人臉偵測（約 0.2 MB）
├── outputs/<job_id>/  # 每次 render 各自一個目錄，不會互相覆蓋；
│                      #   檔名用專案名稱（`EP1 台北_output.mp4`），不是 UUID
└── jobs/<job_id>/     # 分析工作的暫存
```

**模型跟著專案走。** 權重不寫進家目錄的共用快取，而是放在工作區裡，
所以刪掉 `workspace/` 就等於刪掉模型，專案佔多少空間就是它看起來佔的那麼多。
每個模型都釘住 SHA-256，抓下來對不上就整個丟掉並說明理由，不會留下半個檔案
讓下一次誤以為抓好了。要把幾 GB 挪到別的磁碟、或讓多個工作區共用一份，
就設 `CLIP_MCP_MODELS`——代價就是失去上面那個好處。

| 變數 | 預設 | 說明 |
|---|---|---|
| `CLIP_MCP_WORKSPACE` | `workspace` | 工作區路徑 |
| `CLIP_MCP_MODELS` | `<工作區>/models` | 模型權重放哪。預設在工作區內，刪專案就一起刪掉 |
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
部分測試會實際 render 後量測——響度與 ducking、調色、轉場、影音同步、字幕位置、
跟臉裁切有沒有把臉留在畫面上、節拍偵測準不準——所以需要 FFmpeg 在 `PATH` 上。

`src/app/benchmark/` 是另一層：不 render，直接替一份編譯好的剪輯打分——
切到字的剪點、落在壞幀上的剪點、成片長度誤差、必留段落覆蓋率、必剔段落洩漏率，
以及每個剪點還有多少餘裕。全是純函式，所以同一份計畫對同一批素材永遠得到同一個分數。
量不到的東西（例如沒分析過的素材）會被點名，不會混進「沒問題」裡。
語料庫目前只有三支，而且都還是草稿（`corpus/`），所以**現在一分都還算不出來**；
更上層的成片評分也還沒做。這是下一個工作項目，見 [docs/roadmap.md](docs/roadmap.md) §11。

## 架構

各模組職責見 [docs/architecture.md](docs/architecture.md)。
引擎層（`src/app/engine/`）不依賴 MCP，可以單獨使用。
2.0 的方向與里程碑見 [docs/roadmap.md](docs/roadmap.md)。

## 授權

[GNU AGPL v3](LICENSE)
