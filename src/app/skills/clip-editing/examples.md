# Worked examples

Two requests followed through from the user's words to the tool calls and
the reply, for when you want to see the shape of a whole exchange.
Referenced from SKILL.md.

## Joining clips

User: 「把 trip1.mp4 和 trip2.mp4 接起來，trip1 只要 5 到 20 秒，做成直式」

1. `import_asset` on `trip1.mp4` returns `{"id": "a1", "duration": "42.0", "has_audio": true}`.
   `import_asset` on `trip2.mp4` returns `{"id": "a2", "duration": "18.5", "has_audio": true}`.
2. `create_project` with `{"width": 1080, "height": 1920}` returns version 1.
3. `apply_edits`:
   ```json
   {
     "project_id": "<id>",
     "expected_version": 1,
     "operations": [
       {"action": "add_track", "track_id": "main", "track_type": "video"},
       {"action": "insert_clip", "track_id": "main", "clip_id": "trip1", "asset_id": "a1", "source_range": {"start": 5, "end": 20}},
       {"action": "insert_clip", "track_id": "main", "clip_id": "trip2", "asset_id": "a2", "source_range": {"start": 0, "end": 18.5}}
     ]
   }
   ```
4. `render_project` with `is_preview: true`, then poll `get_job` until the job
   completes.
5. Reply: 「預覽好了（33.5 秒，1080x1920 直式，30fps；畫面已裁切填滿）：<path>。
   沒問題的話我再輸出正式版。」

## Adding music

User: 「幫這支加上 song.mp3 當背景音樂」. The project from the example above is
at version 2 with `duration` 33.5.

1. `import_asset` on `song.mp3` returns `{"id": "m1", "duration": "185.2", "has_video": false, "has_audio": true}`.
2. Append the whole song and let `fit_track` deal with the lengths. The video
   clips have sound, so use volume 0.5 and duck the track.
3. `apply_edits` with `expected_version: 2`:
   ```json
   [
     {"action": "add_track", "track_id": "music", "track_type": "audio", "duck_under_speech": true},
     {"action": "insert_clip", "track_id": "music", "clip_id": "song", "asset_id": "m1",
      "source_range": {"start": 0, "end": 185.2}, "volume": 0.5},
     {"action": "fit_track", "track_id": "music", "fade_in": 1, "fade_out": 2}
   ]
   ```
   Note that nothing here needed the video's length: `fit_track` trims the
   song to 33.5 by itself.
4. Render a preview, then reply:
   「加好了：有人講話的時候音樂會自動壓低，空檔時再回來，開頭淡入 1 秒、結尾淡出 2 秒。」
