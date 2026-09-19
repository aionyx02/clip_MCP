Clip-MCP
let user can easy to make professional video



## Tests

```
uv run pytest
```

Tests generate their own footage with FFmpeg and run against a throwaway
workspace, so FFmpeg must be on `PATH`. They never touch `workspace/`.
