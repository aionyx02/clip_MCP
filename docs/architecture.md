# 架構

分三層，由外而內：`server.py` 只負責 MCP 介面與參數驗證，`engine/` 是不依賴 MCP
的影音引擎，`models/` 是純資料與規則。`storage/` 橫跨其上，負責持久化。
引擎層可以單獨拿去用，不必掛在 MCP 底下。

```
clip_MCP/
├── src/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── server.py              # FastMCP 工具入口與路由
│   │   ├── models/                # Pydantic 領域資料模型
│   │   │   ├── __init__.py
│   │   │   ├── media.py           # 素材規格與探測資訊
│   │   │   ├── semantic.py        # 語意片段與語意時間軸（安全剪點、標籤出處、段落選擇）
│   │   │   ├── plan.py            # 剪輯計畫：目標、段落、選材與理由、封閉集合的 trim
│   │   │   ├── timeline.py        # 時間軸、軌道、片段、編輯操作與驗證
│   │   │   └── job.py             # 背景工作與狀態
│   │   ├── engine/                # 影音引擎（不依賴 MCP）
│   │   │   ├── __init__.py
│   │   │   ├── probe.py           # ffprobe 封裝
│   │   │   ├── builder.py         # 時間軸轉譯為 FFmpeg filtergraph（堆疊畫中畫、調色、淡入淡出、響度與 ducking）
│   │   │   ├── ffmpeg.py          # FFmpeg 子程序執行（進度回報、取消）
│   │   │   ├── resources.py       # 實體記憶體量測、各項上限與記憶體估算
│   │   │   ├── analysis.py        # 素材分析：換場、黑畫面、靜止畫面、靜音、逐字稿（faster-whisper）
│   │   │   ├── frames.py          # 抽幀、附時間標籤的縮圖總覽、按輸出比例裁切的 storyboard
│   │   │   ├── semantic.py        # 由分析結果推導 utterance 層語意片段（純函式，不碰模型）
│   │   │   ├── sections.py        # 段落候選邊界的確定性產生，與模型裁決結果的驗證
│   │   │   ├── plan.py            # plan → 時間軸操作的編譯器（純函式），驗證、重新編譯與 diff
│   │   │   ├── subtitles.py       # 逐字稿對應到時間軸的字幕，輸出 ASS（含中文斷行與安全區）
│   │   │   └── renderer.py        # 背景工作排隊准入與 worker 子程序（輸出、分析）
│   │   ├── benchmark/             # L2 評測：不必 render 就能替一份 plan 打分
│   │   │   ├── __init__.py
│   │   │   ├── case.py            # 語料案例：素材 + 指令 + 必留/必剔標註（以相對路徑錨定）
│   │   │   └── metrics.py         # 品質底線的量測（純函式）與計分卡
│   │   ├── storage/               # 專案與任務持久化
│   │   │   ├── __init__.py
│   │   │   └── repo.py            # SQLite 儲存庫（樂觀鎖版本控制，可多行程共用）
│   │   │                          #   assets / projects / jobs / analyses
│   │   │                          #   semantic_timelines / semantic_clips / plans
│   │   └── skills/                # 內建 AI 使用指南（以 MCP resources 提供，不綁定模型）
│   │       └── clip-editing/
│   │           ├── SKILL.md                # 主指南（每次載入）
│   │           ├── pacing-and-structure.md # 剪輯節奏與結構（需要時才讀）
│   │           └── examples.md             # 完整範例（需要時才讀）
│   └── run.py                     # 直接執行用的啟動腳本（等同 clip-mcp 指令）
├── tests/                         # pytest；用 ffmpeg 即時產生素材，工作區隔離在暫存目錄
│   ├── conftest.py                # 共用 fixture：測試用工作區與合成影音檔
│   ├── helpers.py                 # 建立專案與編輯操作的輔助函式
│   ├── test_frames.py             # 時間格式、裁切比例、縮圖拼貼
│   ├── test_import_folder.py      # 資料夾匯入：自然排序、跳過壞檔、ID 穩定
│   ├── test_preview_project.py    # storyboard：每段保底、空隙、音軌、封頂、錯誤
│   ├── test_edit_operations.py    # 切開與重排（含倒敍）
│   ├── test_render_audio.py       # 實際 render 後量測響度與 ducking
│   ├── test_render_look.py        # 實際 render 後量測調色與淡入淡出
│   ├── test_subtitles.py          # 字幕對應、斷行、燒錄位置
│   ├── test_picture_in_picture.py # 疊加時機、邊界、聲音混入
│   ├── test_fit_track.py          # 音樂對齊影片長度（裁切、接續、循環、淡出搬家）
│   ├── test_job_limits.py         # 記憶體閘門：排隊順序、記憶體判準、逾時釋放、解碼器上限
│   ├── test_semantic_timeline.py  # 語意時間軸：切分規則、安全剪點、分數、釘住、查詢、標籤出處
│   ├── test_sections.py           # 段落候選訊號、只能在候選裡裁決、覆蓋性驗證
│   ├── test_edit_plan.py          # plan 編譯的秒數、拒絕條件、diff、工具層往返
│   ├── test_skill_resources.py    # skill 與 server 不得漂移（工具、操作、分層指標）
│   ├── test_benchmark.py          # L2 計分：案例驗證、五項底線、剪點餘裕、端到端
│   └── test_end_to_end.py         # 從資料夾到成片的完整流程
├── docs/
│   ├── architecture.md
│   └── roadmap.md                 # 只留待辦：還缺的量測與剪輯功能、評測、里程碑
├── workspace/                     # 本機資料與產出（CLIP_MCP_WORKSPACE 可覆寫，已 gitignore）
│   ├── clip_mcp.db                # 專案、素材、分析結果、工作狀態
│   ├── outputs/<job_id>/          # 每次 render 各自一個目錄，成品與字幕 ASS 都在裡面
│   └── jobs/<job_id>/             # 分析工作的暫存
├── .mcp.json                      # 客戶端設定：在此目錄啟動的 MCP 客戶端會自動掛上本伺服器
├── fastmcp.json                   # 伺服器設定：fastmcp CLI 的進入點與環境（fastmcp run / dev）
├── pyproject.toml                 # 相依套件與 clip-mcp 進入點（uv 管理，鎖在 uv.lock）
└── README.md
```

## 幾個貫穿全域的約定

- **時間單位**：模型與工具之間一律用秒（float）。`source_range` 是來源檔的秒數，
  `timeline_in` 是剪輯後時間軸的秒數。
- **樂觀鎖**：`apply_edits` 必須帶 `expected_version`。版本不符就整批不套用，
  呼叫端重讀 `get_project` 再重試。失敗的呼叫不會留下半套狀態。
- **語意時間軸**：`engine/semantic.py` 把 `MediaAnalysis` 推導成語意片段，是一支
  **純函式**——句界、換場、靜音都已經在分析結果裡，所以不碰模型、不抽樣，
  同一份分析永遠得到同一批片段。`safe_in` / `safe_out` 是這層的重點：它把
  「剪這裡會不會切到字」從模型的判斷變成量出來的資料。
  建置以輸入雜湊釘住（涵蓋各素材分析內容與 `DERIVATION_VERSION`，但不含
  `analyzed_at`），timeline id 與 clip id 都由該雜湊推導，所以同一批素材在任何機器上
  都得到同一組 ID；重建不變的輸入會原樣取回既有建置，引用不會失效。
- **段落裁決**：`section` 不能純靠規則找出來，但可以縮小範圍。`engine/sections.py`
  用實測訊號（停頓長度、語彙翻新程度、換場密集度、話語標記）**過量產生**候選邊界，
  模型只能在候選裡挑選與命名，伺服器驗證的是「回來的 ID 在不在清單裡」以及
  「這些段落有沒有把素材蓋滿」——是成員資格檢查，不是解讀。
  模型最糟的錯誤因此是段落分得不好，**不可能是切到字**，因為每個候選本來就長在
  實測的靜音邊緣上。候選的權重與門檻是暫定值，記在 timeline 的 `candidate_hash` 裡，
  之後靠語料調校會產生一份明顯不同的新建置，不會把已經談定的段落底下的邊界悄悄挪走。
- **topic 是標籤不是層級**：主題可以不連續、可以跨檔案，沒有自己的區間可以當成 clip，
  所以它是 section 上的一個 `topic` 欄位，同一個標籤在不同檔案的 section 上就把它們串起來。
- **plan 編譯器**：`engine/plan.py` 是這套設計賴以成立的確定性那一半。模型決定選哪些片段、
  為什麼；從那裡開始的每一件事——剪點落在哪、留多少呼吸、哪些併成一段、成片多長——
  都是 plan 與語意時間軸的純函式。`trim` 是**封閉集合**（`full` / `keep` / `head` /
  `tail` / `tighten`），沒有自然語言指令：編譯器一旦需要解讀句子就得在裡面塞一個模型，
  同一份 plan 就不再決定同一個結果。驗證不過的 plan 原樣退回並附理由，不默默修正。
- **出身與鎖定**：`compile_plan` 產生的每個 clip 都記下 `from_plan_id` 與
  `from_clip_ids`（它是由哪些語意片段合併而成的）。出身不經過工具介面——
  操作套用完之後由 `_apply` 的 stamp 直接寫上去，所以呼叫端碰不到它。
  **手改就是鎖定**：`trim_clip` / `move_clip` / `split_clip` / `reorder_clip` /
  `set_clip_look` / `set_clip_audio` 動到一個有出身的 clip，它就自動 `pinned`，
  沒有人需要記得多下一個指令——忘記的代價是回饋迴圈每跑一次就洗掉使用者的工作。
  重新編譯時，pinned 的 clip 連同它被改過的 source_range 與設定原樣保留，
  只有在序列裡的位置由新的 plan 決定；配對靠的是 `from_clip_ids` 完全相同。
  如果新 plan 已經不再用到某個 pinned clip 的素材，**整個編譯被拒絕並列出擋路的 clip**——
  丟掉使用者的工作比停下來糟。編譯只擁有主序列與音樂軌，其他軌道原封不動。
- **背景工作**：render 與分析都跑在獨立 worker 子程序，狀態寫進 SQLite，
  所以伺服器重啟不會中斷工作，也能多行程共用同一個工作區。
- **記憶體閘門**：工作先排隊，不是呼叫就跑。`JobManager._plan` 在
  `Repository.update_active_jobs` 的單一 immediate transaction 內，一次看完所有未完成
  的工作再決定放行誰——所以多個伺服器或 worker 不會各自算出「還能再開一個」。
  判準有兩個：同時執行數（`CLIP_MCP_MAX_JOBS`）與剩餘實體記憶體減去保留量
  （`CLIP_MCP_MEMORY_RESERVE_MB`）。剛放行的工作還沒真的配置記憶體，所以 `WARMUP`
  之內它的估算值會額外從可用量扣除。排隊依先到先服務，不讓小工作一直插隊；
  但沒有任何工作在跑時，隊首一定放行——全部拒絕比慢慢跑更糟。
  worker 結束與 `get_job` 都會再推一次隊伍，所以不需要另外的排程器。
- **評測分層**：L1 是測試套件（算術對不對），L3 是判官評成片（還沒做）。
  `benchmark/` 是中間的 L2：**不 render**，直接拿編譯後的 windows 對語料的標註算分。
  五項底線（切到字、落在壞幀、長度誤差、必留覆蓋、必剔洩漏）全是純函式，
  所以同一份 plan 對同一批素材永遠得到同一個分數——分數掉了才算得上是證據。
  標註一律以**相對路徑 + 該檔秒數**錨定，不用 asset ID：asset ID 是每個工作區各自產生的，
  用它當錨點的語料換一台機器就什麼都量不到。
  沒分析過的素材不會被當成「沒問題」——`missing_analyses` 會把它點名出來，
  因為「沒量到」和「量了沒事」在計分卡上長得一模一樣。
- **skill 與程式碼同步**：`test_skill_resources.py` 雙向比對——每個工具與編輯操作都要在
  `SKILL.md` 裡提到，指南也不能提到不存在的操作，避免指南跟實作漂移。
