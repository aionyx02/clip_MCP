---
name: clip-editing
description: Turn natural-language video editing requests (for example 「把這三支影片接起來」, 「剪掉講錯的地方」, 「加背景音樂」, "move the last clip to the front") into clip-mcp tool calls. Use it whenever the user wants to understand, cut, join, trim, reorder, add music to, or render videos with this server, including when the request depends on what is said or shown in the footage, or is vague or missing details.
---

# Clip editing

This server edits videos on a timeline and renders MP4 files with FFmpeg.
Follow this guide to translate what the user asks for into tool calls. Reply
to the user in their language.

## What is supported today

Supported:
- Cutting segments out of video files and arranging them in sequence on one
  video track.
- Trimming, deleting, inserting, reordering, and replacing clips.
- Black-and-silent gaps between clips.
- Background music on audio tracks, from MP3, M4A, WAV, or the sound of
  another video, with per-clip volume and fade-in and fade-out. This includes
  muting or lowering a video's original sound.
- Mixed sources: different resolutions, orientations, and frame rates, and
  videos without sound. Every source is scaled to fill the output size and
  center-cropped.
- Fast 480p previews and full-quality renders in the background, with
  progress.
- Understanding footage: scene changes, black and frozen picture, silences,
  and a transcript with accurate word timings (`analyze_asset`,
  `get_analysis`), plus labeled frame contact sheets (`view_frames`).
  Everything runs locally; nothing is uploaded.

Not supported yet: more than one video track (picture-in-picture, overlays),
automatically lowering music while someone speaks (ducking), still images,
speed changes, transitions, text or subtitles, and filters or effects. When a
request needs one of these, say so plainly, offer the closest supported
result, and never pretend it was done. For example:
「目前還不支援人聲出現時自動壓低音樂。我先把音樂整體調到 0.25，讓人聲聽得清楚，可以嗎？」

## Concepts

- **Asset**: a media file registered with `import_asset`. Its `duration` is
  the source length in seconds. `has_video` and `has_audio` tell you which
  kind of track it can go on.
- **Project**: output `width`, `height`, and frame rate, plus its tracks.
  These settings cannot be changed later; create a new project instead.
  `get_project` also returns `duration`, the length of the edited video.
- **Tracks**: one `video` track holds the picture and each clip's own sound.
  Any number of `audio` tracks hold background music. The video track
  decides the output length; audio past the end of the video is cut off.
- **Clip**: the part of an asset between `source_range.start` and
  `source_range.end` (seconds in the source file), placed on a track at
  `timeline_in`. Its length on the timeline is `end - start`. It also has
  `volume` (1.0 is the original level) and `audio_fade_in` /
  `audio_fade_out` in seconds.
- **Version**: `apply_edits` needs the project's current `version` as
  `expected_version`. A failed call changes nothing, so fix the operations and
  retry with the same version.
- **Magnetic editing**: `insert_clip` pushes later clips on the same track
  back to make room. `trim_clip` and `delete_clip` pull later clips on the
  same track forward to close gaps, unless `ripple` is false. Only `add_clip`
  and `move_clip` use absolute positions, and they never move other clips.
  Clips on other tracks never move, so music stays where it is when you edit
  the video. Prefer the magnetic operations so you never have to calculate
  positions yourself.

## Workflow

1. Read the request and settle anything missing; see
   [When the request is incomplete](#when-the-request-is-incomplete).
2. Call `import_asset` for every source file, including music. Use the
   returned `duration` instead of guessing lengths. Use `inspect_media` if
   you need resolution or frame rate.
3. Call `create_project` with the output settings; see
   [Defaults](#defaults). Then call `apply_edits` with an `add_track`
   operation, `{"action": "add_track", "track_id": "main", "track_type": "video"}`,
   together with the first clips.
4. Build the video sequence with `insert_clip`. Without `before_clip_id`,
   each clip is appended after the last one. Give clips short, meaningful IDs
   such as `intro`, `beach`, or `c1`.
5. When the edit depends on what is said or shown, read the footage first;
   see [Understanding footage](#understanding-footage).
6. Add music, if requested, as described in
   [Background music](#background-music).
7. Put all operations for one request into a single `apply_edits` call.
8. Call `render_project` with `is_preview: true` unless the user asked for the
   final file. Poll `get_job` every few seconds until the status is
   `completed`, `failed`, or `cancelled`. Report the output path and length.
9. Render the full-quality version with `is_preview: false` once the user is
   happy, or immediately if they asked for the final file.

For later edits, call `get_project` first so your operations match the current
clips and version.

## Understanding footage

Use this whenever a request depends on content: what someone says, what is on
screen, where the pauses or mistakes are, or which parts are best.

1. Call `analyze_asset` for each source. Set `language` when you know it. For
   Chinese speech, set `chinese_variant` to match how the user writes:
   `zh-TW` for Traditional Chinese as used in Taiwan (the default for users
   who write Traditional Chinese), `zh-HK` for Hong Kong, or `zh-Hans` for
   Simplified. Put names or terms the speaker uses into `prompt`. The first
   run downloads the speech model. Poll `get_job`; for long files, tell the
   user the current `stage` and `progress`.
2. Read `get_analysis` in windows of about 10 minutes (`start` / `end`). All
   times are seconds in the source file, so they can be used directly as
   `source_range`.
3. Look before you cut. Call `view_frames` with about 12 frames over the whole
   asset for an overview, then use a narrow range to check a specific moment.
   Do not describe what is on screen without looking.
4. Turn the results into clips:
   - Use transcript segment `start` / `end` as `source_range`. Add about
     0.1 s before and 0.15 s after so words are not clipped, but stay inside
     the neighboring `silences` and the asset duration.
   - Merge kept parts that are less than 0.3 s apart into one clip, so the
     result is not choppy.
   - Build the sequence with `insert_clip` in source order.
   - Use `include_words: true` only when you need to cut in the middle of a
     sentence.
5. For subjective selections such as highlights, list the chosen parts with
   times and quoted text, and confirm with the user before rendering.

Transcripts can misrecognize names and jargon. When a quote matters, check the
surrounding segments, and ask the user if the meaning is unclear.

## Background music

1. Add an audio track, such as
   `{"action": "add_track", "track_id": "music", "track_type": "audio"}`, and
   append the song with `insert_clip`. To start the song at a later point,
   such as the chorus, set `source_range.start` to that point. To start the
   music later in the video, use `add_clip` with `timeline_in` instead.
2. Make it cover the video. Compare the music length with the project
   `duration` once all video edits are in the batch.
   - **Music shorter than the video:** append the song again with
     `insert_clip` until the video is covered.
   - **Music longer than the video:** trim the last music clip so it ends
     exactly at the video end: `end = start + (duration - timeline_in)`.
     Otherwise the music is cut off abruptly.
3. Set levels and fades with `insert_clip` fields or `set_clip_audio`:
   - Music volume: see [Defaults](#defaults).
   - `audio_fade_in: 1` on the first music clip.
   - `audio_fade_out: 2` on the last music clip.
4. Whenever a later edit changes the video `duration`, repeat step 2 and move
   the fade-out to the new last music clip.

## Mapping requests to operations

Times like `1:30` mean 90 seconds. Unless the user refers to the edited result,
times refer to the source file.

| Request | Operations |
|---|---|
| 「把 a.mp4、b.mp4 接起來」 / join these videos | `insert_clip` for each file in order, `source_range` `{start: 0, end: duration}` |
| 「a.mp4 只要 10 到 20 秒」 / use 0:10–0:20 of a.mp4 | `insert_clip` with `source_range` `{start: 10, end: 20}` |
| 「剪掉 X 開頭 3 秒」 / cut the first 3 s of X | `trim_clip` X, `new_source_range` `{start: start + 3, end: end}` |
| 「剪掉 X 最後 3 秒」 / cut the last 3 s of X | `trim_clip` X, `new_source_range` `{start: start, end: end - 3}` |
| 「刪掉第二段」 / delete the second clip | `delete_clip` (later clips move up) |
| 「刪掉 X 但留黑畫面」 / remove X but keep the gap | `delete_clip` X with `ripple: false` |
| 「在 A 和 B 中間插入 c.mp4」 / insert between A and B | `insert_clip` with `before_clip_id: B` |
| 「把 C 移到最前面」 / move C to the front | `delete_clip` C, then `insert_clip` with C's `asset_id` and `source_range`, `clip_id` C, `before_clip_id` = first clip |
| 「把 A 移到最後」 / move A to the end | `delete_clip` A, then `insert_clip` A without `before_clip_id` |
| 「把 A 換成 d.mp4」 / replace A with d.mp4 | `insert_clip` new clip with `before_clip_id: A`, then `delete_clip` A |
| 「A 和 B 中間停 2 秒黑畫面」 / 2 s of black before B | `move_clip` B and every later video clip to `timeline_in + 2` |
| 「加背景音樂 song.mp3」 / add background music | See [Background music](#background-music) |
| 「音樂從第 10 秒才進來」 / start the music at 0:10 | `move_clip` the first music clip to `timeline_in: 10` (or `add_clip` there) |
| 「用歌的 1:05 開始」 / start from the chorus at 1:05 | Music clip `source_range.start: 65` |
| 「音樂小聲一點／大聲一點」 / music quieter or louder | `set_clip_audio` on every music clip, `volume` × 0.6 or × 1.5 |
| 「把影片原音關掉」 / mute the original sound | `set_clip_audio` `volume: 0` on every video clip |
| 「音樂淡出」 / fade the music out | `set_clip_audio` `audio_fade_out: 2` on the last music clip, after trimming it to end at the video end |
| 「拿掉背景音樂」 / remove the music | `delete_clip` every clip on the audio track |
| 「做成直式／方形」 / make it vertical or square | New project with the new size, then rebuild the clips |
| 「這支影片在講什麼」 / what is this video about | `analyze_asset`, then summarize `get_analysis` with times, and `view_frames` for the visuals |
| 「剪掉講錯／重講的地方」 / remove flubbed takes | From the transcript, keep only the last complete version of each repeated sentence |
| 「去掉停頓／氣口」 / remove pauses | Keep the speech between `silences` longer than about 0.6 s, with padding, as consecutive clips |
| 「只留有講到 X 的段落」 / keep only parts about X | Keep the transcript segments about X, plus enough context to make sense |
| 「剪掉黑畫面／畫面卡住的地方」 / remove black or frozen parts | Keep the time outside `black_frames` / `frozen_frames` |
| 「找出精華剪成 60 秒」 / a 60 s highlight reel | Choose segments by transcript and frames until about 60 s; confirm the list before rendering |
| 「第 3 分鐘那個畫面是什麼」 / what is on screen at 3:00 | `view_frames` with `start: 175`, `end: 185`, `count: 4` |

"The second clip" means the second video clip ordered by `timeline_in`. For a
time in the edited result, such as 「成片第 12 秒」, find the clip where
`timeline_in <= 12 < timeline_in + (end - start)`. The source time is
`start + (12 - timeline_in)`.

## Defaults

Use these without asking, and mention the ones you chose in one short line.

| Setting | Default |
|---|---|
| Output size | 1080x1920 for 直式, 短影音, Reels, Shorts, 抖音, or TikTok; 1080x1080 for 方形 or IG 貼文; 1920x1080 otherwise, or 1080x1920 if most sources are vertical |
| Frame rate | 30 (`fps_num: 30, fps_den: 1`). Use 25, 60, or 29.97 (`30000/1001`) only if asked |
| Clip order | The order the user listed files in; otherwise file-name order |
| Music volume | 0.25 if any video clip has sound (`has_audio`), so speech stays clear; 0.8 if no video clip has sound |
| Music fades | 1 s fade-in on the first music clip, 2 s fade-out ending at the video end |
| Render | Preview first |

## When the request is incomplete

Ask only when a wrong guess would waste a render or go against what the user
wants:
- No source file is named, or several files could match. This includes
  requests like 「加點音樂」: there is no built-in music library, so ask for a
  music file.
- It is unclear which part to keep or remove, as in 「剪短一點」 or
  「剪精華」. Propose concrete choices, for example
  「要保留前 30 秒，還是去掉開頭和結尾各 5 秒？」
- The order of several clips is unclear and cannot be inferred.

Do not ask about anything you can look up, such as durations
(`import_asset`, `inspect_media`), the current timeline (`get_project`), or
what is said or shown in the footage (`analyze_asset`, `get_analysis`,
`view_frames`).
Do not ask about [Defaults](#defaults). Put all open questions into one short
message with options the user can pick from.

## Handling errors

| Error | What to do |
|---|---|
| `version conflict` | Call `get_project`, re-plan against the latest clips, and retry. |
| `overlaps clip` | Use `insert_clip` or rippling operations instead of absolute positions. |
| `source range ends at ... but asset ... is only ...` | Clamp `end` to the asset duration and tell the user. |
| `has no audio stream, so it cannot be used on audio track` | That file has no sound; ask the user for another music file. |
| `put audio-only assets on an audio track` | Move the file to an audio track. |
| `audio fades ... are longer than the clip` | Shorten the fades to fit the clip. |
| `shorter than one frame` | Drop the clip or make it longer. |
| `no clips on a video track to render` | Add video clips first; the server cannot render music alone. |
| `has not been analyzed; call analyze_asset first` | Run `analyze_asset`, wait for the job, then retry. |
| `has no video frames to show` | The asset is audio-only or a still image; use `get_analysis` instead. |
| `... is not supported yet` | Explain the limit and offer the closest supported result. |
| Job `failed` | Summarize `error_message` for the user. Do not retry the same render unchanged. |

## Examples

### Joining clips

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

### Adding music

User: 「幫這支加上 song.mp3 當背景音樂」. The project from the example above is
at version 2 with `duration` 33.5.

1. `import_asset` on `song.mp3` returns `{"id": "m1", "duration": "185.2", "has_video": false, "has_audio": true}`.
2. The song is longer than the video, so trim it to end at 33.5.
   The video clips have sound, so use volume 0.25.
3. `apply_edits` with `expected_version: 2`:
   ```json
   [
     {"action": "add_track", "track_id": "music", "track_type": "audio"},
     {"action": "insert_clip", "track_id": "music", "clip_id": "song", "asset_id": "m1",
      "source_range": {"start": 0, "end": 33.5}, "volume": 0.25, "audio_fade_in": 1, "audio_fade_out": 2}
   ]
   ```
4. Render a preview, then reply:
   「加好了：音樂音量 0.25（保留原本的人聲），開頭淡入 1 秒、結尾淡出 2 秒。」
