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
│   │   └── renderer.py        # 子程序管理與執行
│   └── storage/               # 專案與任務持久化
│       ├── __init__.py
│       └── repo.py            # SQLite / 記憶體儲存庫（樂觀鎖版本控制）
├── workspace/                 # 本機素材、暫存與成片目錄
│   ├── uploads/
│   ├── previews/
│   └── outputs/
├── requirements.txt
└── run.py                     # 啟動腳本