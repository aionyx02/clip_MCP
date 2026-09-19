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
│   │   │   ├── timeline.py        # 時間軸、軌道、片段、編輯操作與驗證
│   │   │   └── job.py             # 背景工作與狀態
│   │   ├── engine/                # 影音引擎（不依賴 MCP）
│   │   │   ├── __init__.py
│   │   │   ├── probe.py           # ffprobe 封裝
│   │   │   ├── builder.py         # 時間軸轉譯為 FFmpeg filtergraph（堆疊畫中畫、調色、淡入淡出、響度與 ducking）
│   │   │   ├── ffmpeg.py          # FFmpeg 子程序執行（進度回報、取消）
│   │   │   ├── analysis.py        # 素材分析：換場、黑畫面、靜止畫面、靜音、逐字稿（faster-whisper）
│   │   │   ├── frames.py          # 抽幀、附時間標籤的縮圖總覽、按輸出比例裁切的 storyboard
│   │   │   ├── subtitles.py       # 逐字稿對應到時間軸的字幕，輸出 ASS（含中文斷行與安全區）
│   │   │   └── renderer.py        # 背景工作管理與 worker 子程序（輸出、分析）
│   │   ├── storage/               # 專案與任務持久化
│   │   │   ├── __init__.py
│   │   │   └── repo.py            # SQLite 儲存庫（樂觀鎖版本控制，可多行程共用）
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
│   ├── test_skill_resources.py    # skill 與 server 不得漂移（工具、操作、分層指標）
│   └── test_end_to_end.py         # 從資料夾到成片的完整流程
├── docs/
│   ├── architecture.md
│   └── roadmap.md                 # 2.0 方向：語意時間軸、剪輯規劃器、評測與里程碑
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
- **背景工作**：render 與分析都跑在獨立 worker 子程序，狀態寫進 SQLite，
  所以伺服器重啟不會中斷工作，也能多行程共用同一個工作區。
- **skill 與程式碼同步**：`test_skill_resources.py` 雙向比對——每個工具與編輯操作都要在
  `SKILL.md` 裡提到，指南也不能提到不存在的操作，避免指南跟實作漂移。
