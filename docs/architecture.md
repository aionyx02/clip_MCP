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
│   │   ├── builder.py         # 時間軸轉譯為 FFmpeg Filtergraph
│   │   ├── ffmpeg.py          # FFmpeg 子程序執行（進度回報、取消）
│   │   ├── analysis.py        # 素材分析：換場、黑畫面、靜止畫面、靜音、逐字稿（faster-whisper）
│   │   ├── frames.py          # 抽幀與附時間標籤的縮圖總覽
│   │   └── renderer.py        # 背景工作管理與 worker 子程序（輸出、分析）
│   ├── storage/               # 專案與任務持久化
│   │   ├── __init__.py
│   │   └── repo.py            # SQLite 儲存庫（樂觀鎖版本控制，可多行程共用）
│   └── skills/                # 內建 AI 使用指南（以 MCP resources 提供，不綁定模型）
│       └── clip-editing/
│           └── SKILL.md
├── workspace/                 # 本機素材、暫存與成片目錄（CLIP_MCP_WORKSPACE 可覆寫）
│   ├── clip_mcp.db            # 專案、素材、工作狀態
│   ├── uploads/
│   ├── previews/
│   └── outputs/
├── requirements.txt
└── run.py                     # 啟動腳本