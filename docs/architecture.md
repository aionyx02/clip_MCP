src/
├── app/
│   ├── __init__.py
│   ├── server.py              # FastMCP 工具入口與路由
│   ├── models/                # Pydantic 領域資料模型
│   │   ├── __init__.py
│   │   ├── media.py           # 素材規格與探測資訊
│   │   ├── timeline.py        # 時間軸、軌道、片段、操作
│   │   └── job.py             # 背景工作與狀態
│   ├── engine/                # 影音引擎（無依賴 MCP）
│   │   ├── __init__.py
│   │   ├── probe.py           # ffprobe 封裝
│   │   ├── builder.py         # 時間軸轉譯為 FFmpeg Filtergraph（堆疊畫中畫、調色、淡入淡出、響度與 ducking）
│   │   ├── ffmpeg.py          # FFmpeg 子程序執行（進度回報、取消）
│   │   ├── analysis.py        # 素材分析：換場、黑畫面、靜止畫面、靜音、逐字稿（faster-whisper）
│   │   ├── frames.py          # 抽幀、附時間標籤的縮圖總覽、按輸出比例裁切的 storyboard
│   │   ├── subtitles.py       # 逐字稿對應到時間軸的字幕，輸出 ASS（含中文斷行與安全區）
│   │   └── renderer.py        # 背景工作管理與 worker 子程序（輸出、分析）
│   ├── storage/               # 專案與任務持久化
│   │   ├── __init__.py
│   │   └── repo.py            # SQLite 儲存庫（樂觀鎖版本控制，可多行程共用）
│   └── skills/                # 內建 AI 使用指南（以 MCP resources 提供，不綁定模型）
│       └── clip-editing/
│           ├── SKILL.md                # 主指南（每次載入）
│           ├── pacing-and-structure.md # 剪輯節奏與結構（需要時才讀）
│           └── examples.md             # 完整範例（需要時才讀）
├── tests/                      # pytest；用 ffmpeg 即時產生素材，工作區隔離在暫存目錄
│   ├── conftest.py             # 共用 fixture：測試用工作區與合成影音檔
│   ├── helpers.py              # 建立專案與編輯操作的輔助函式
│   ├── test_frames.py          # 時間格式、裁切比例、縮圖拼貼
│   ├── test_import_folder.py   # 資料夾匯入：自然排序、跳過壞檔、ID 穩定
│   ├── test_preview_project.py # storyboard：每段保底、空隙、音軌、封頂、錯誤
│   ├── test_edit_operations.py # 切開與重排（含倒敍）
│   ├── test_render_audio.py    # 實際 render 後量測響度與 ducking
│   ├── test_render_look.py     # 實際 render 後量測調色與淡入淡出
│   ├── test_subtitles.py       # 字幕對應、斷行、燒錄位置
│   ├── test_picture_in_picture.py # 疊加時機、邊界、聲音混入
│   ├── test_fit_track.py       # 音樂對齊影片長度（裁切、接續、循環、淡出搬家）
│   ├── test_skill_resources.py # skill 與 server 不得漂移（工具、操作、分層指標）
│   └── test_end_to_end.py      # 從資料夾到成片的完整流程
├── workspace/                 # 本機素材、暫存與成片目錄（CLIP_MCP_WORKSPACE 可覆寫）
│   ├── clip_mcp.db            # 專案、素材、工作狀態
│   ├── uploads/
│   ├── previews/
│   └── outputs/
├── requirements.txt
└── run.py                     # 啟動腳本