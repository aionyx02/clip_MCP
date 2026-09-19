---
name: clip-editing
description: Turn natural-language video editing requests (for example 「把這三支影片接起來」, 「剪掉講錯的地方」, 「加背景音樂」, "move the last clip to the front") into clip-mcp tool calls. Use it whenever the user wants to understand, cut, join, trim, reorder, add music to, or render videos with this server, including when the request depends on what is said or shown in the footage, when the user points at a whole folder of unedited footage without saying what to make of it, or when the request is vague or missing details.
---

# Clip editing

This server edits videos on a timeline and renders MP4 files with FFmpeg.
Follow this guide to translate what the user asks for into tool calls. Reply
to the user in their language.

## What is supported today

Supported:
- Cutting segments out of video files and arranging them in sequence on one
  video track.
- Trimming, splitting, deleting, inserting, reordering, and replacing clips,
  including non-linear orders such as opening with the ending.
- Black-and-silent gaps between clips.
- Background music on audio tracks, from MP3, M4A, WAV, or the sound of
  another video, with per-clip volume and fade-in and fade-out. This includes
  muting or lowering a video's original sound.
- Ducking: an audio track marked `duck_under_speech` drops automatically
  while the video's own sound is loud, so music stays under the talking and
  comes back up in the gaps.
- One consistent output level. Every render is normalized to a loudness
  target, -14 LUFS by default, so footage recorded on different devices does
  not jump from clip to clip and the file matches what streaming platforms
  expect.
- Mixed sources: different resolutions, orientations, and frame rates, and
  videos without sound. Every source is scaled to fill the output size and
  center-cropped.
- Fades from and to black, per clip, on the picture as well as the sound. Two
  clips can dip through black between them without changing any timing.
- Colour per clip: brightness, contrast, saturation, and white balance in
  kelvin, including turning a clip black and white.
- Picture in picture: video tracks above the first are drawn on top of it,
  each clip in a box given as fractions of the frame.
- Burned-in captions from the transcript, moved onto the edited timeline,
  broken between words, wrapped to fit, and kept clear of the controls a
  phone draws over a vertical video (`generate_subtitles`, then a
  `set_subtitles` operation, then `render_project` with `burn_subtitles`).
- Storyboards of the edited sequence in seconds (`preview_project`), plus
  fast 480p previews and full-quality renders in the background, with
  progress.
- Understanding footage: scene changes, black and frozen picture, silences,
  and a transcript with accurate word timings (`analyze_asset`,
  `get_analysis`), plus labeled frame contact sheets (`view_frames`).
  Everything runs locally; nothing is uploaded.

Not supported yet: still images, speed changes, cross dissolves and wipes
between clips (only fades through black), sound leading or trailing the
picture (J and L cuts), free text and graphics other than captions, and
filters beyond the colour controls above. When a request needs one of these,
say so plainly, offer the closest supported result, and never pretend it was
done. For example:
「目前還做不出兩段重疊淡入淡出的疊化。我可以讓前一段淡出到黑、後一段再淡入，看起來也很自然，可以嗎？」

## Concepts

- **Asset**: a media file registered with `import_asset`. Its `duration` is
  the source length in seconds. `has_video` and `has_audio` tell you which
  kind of track it can go on.
- **Project**: output `width`, `height`, and frame rate, plus its tracks.
  These settings cannot be changed later; create a new project instead.
  `get_project` also returns `duration`, the length of the edited video.
- **Tracks**: the first `video` track is the base: it holds the sequence and
  each clip's own sound, and it decides the output length. Further `video`
  tracks are drawn on top of it, for picture in picture; their clips carry
  their own sound too. Any number of `audio` tracks hold background music.
  Anything past the end of the base track is cut off.
- **Layout**: a clip on a video track above the base is drawn in the box its
  `layout` gives, as fractions of the frame: `{"x": 0.62, "y": 0.06,
  "width": 0.32, "height": 0.32}` is a small inset in the top right. Fractions
  rather than pixels, so the same box works at any output size. A clip with no
  layout covers the whole frame, which is how you cut away to another shot. A
  `layout` only means something above the base track, so putting one on a base
  clip is refused rather than quietly ignored.
- **Captions**: `generate_subtitles` proposes captions from the transcripts of
  the base-track clips that survived the edit, already moved onto the
  timeline; insets on higher tracks are pictures over that sound, so they are
  not captioned. Nothing is saved until you apply them with `set_subtitles`,
  so check the wording first, and they only appear in the file when
  `render_project` is called with `burn_subtitles`. Stored captions belong to
  the cut they were made from: they do not move when clips do, so caption
  last, and generate them again after any later change to the cut.
  `get_project` reports how many are stored as `subtitle_count`.
- **Clip**: the part of an asset between `source_range.start` and
  `source_range.end` (seconds in the source file), placed on a track at
  `timeline_in`. Its length on the timeline is `end - start`. It also has
  `volume` (1.0 is the original level) and `audio_fade_in` /
  `audio_fade_out` in seconds.
- **Version**: `apply_edits` needs the project's current `version` as
  `expected_version`. `get_project` leaves out clip fields that are still at
  their default, so a clip with no `volume` is at 1.0 and one with no `color`
  is as shot. It leaves captions out and reports only how many there are as
  `subtitle_count`; read them with `get_subtitles`, a window at a time, or
  pass `include_subtitles: true` to get the whole set back with the project. A failed call changes nothing, so fix the operations and
  retry with the same version.
- **Magnetic editing**: `insert_clip` pushes later clips on the same track
  back to make room. `trim_clip` and `delete_clip` pull later clips on the
  same track forward to close gaps, unless `ripple` is false. `reorder_clip`
  moves a clip to another place in the sequence in one step, closing the
  space it leaves and opening space where it lands; the clip keeps its cut,
  volume, and fades, so use it instead of deleting and re-inserting, which
  makes you restate the source range and risks changing the picture.
  `split_clip` cuts a clip in two where you ask, by `at` on the timeline or
  `at_source` in the source file, and moves nothing. Both halves keep the
  clip's colour and box, and its fades stay on the outer edges of the shot
  rather than landing on the new cut. Only `add_clip` and
  `move_clip` use absolute positions, and they never move other clips.
  Clips on other tracks never move, so music stays where it is when you edit
  the video. Prefer the magnetic operations so you never have to calculate
  positions yourself.

## Talking to the user

The person using this server is editing their own video, not writing code.
Write to them as an editor would, not as a program reporting its state.

- Lead with the result: what was made, how long it is, and the file path.
  Then one short line on what you decided for them. Nothing else.
- Keep replies to three or four lines. Say what changed, not how.
- Never put tool names, asset or clip IDs, JSON, or version numbers in a
  reply, and translate errors instead of quoting them: not
  「apply_edits version conflict」, but
  「我對了一下目前的順序，已經接好了」.
- Write times as 0:12 or 1:05, and lengths in plain words such as
  「33 秒」.
- Give two or three concrete options to choose from instead of an open
  question such as 「你想怎麼剪？」.

**Read the plan back before anything expensive.** If the plan involves a
length or a structure you chose, read
`skill://clip-editing/pacing-and-structure.md` first, so the sentence you read
back is one worth agreeing to. A render or an `analyze_asset` costs minutes, while `preview_project` takes seconds, so
before the first `render_project` of a request, and before analyzing a file
longer than about ten minutes, restate the plan in one sentence in plain
language and wait for a yes. When the edit is already built, send the
storyboard with it; one image settles more than a paragraph:

「我理解成：trip1.mp4 和 trip2.mp4 接起來、trip1 只要 5 到 20 秒，做成直式。先出預覽？」

Do this once per request, not once per operation, and skip it when the user
has already approved this exact plan. The readback is also what lets you stop
asking about details: state your assumption in it, and the user corrects what
is wrong before a render is wasted.

## Workflow

1. Read the request and settle anything missing; see
   [When the request is incomplete](#when-the-request-is-incomplete). When the
   user brings unedited footage and no plan, start from
   [When the user brings raw footage and no plan](#when-the-user-brings-raw-footage-and-no-plan)
   instead.
2. Call `import_asset` for every source file, including music, or
   `import_folder` when the user points at a folder, with `recursive: true`
   when the footage sits in sub-folders. Use the returned `duration` instead
   of guessing lengths. Use `inspect_media` if you need resolution or frame
   rate.
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
8. If the user asked for captions, call `generate_subtitles`, show them the
   lines with their times, and store them with `set_subtitles` once they are
   happy. Transcripts mishear names, so this is worth reading rather than
   applying blind. Afterwards, correct single lines with `edit_subtitle` and
   its `cue_id`; never resend the whole set to change one word. Lower
   `max_characters` or `max_seconds` if the user wants shorter lines on
   screen. Caption last, after the cut is settled: stored captions do not
   move when clips do, so any later edit that changes the cut means calling
   `generate_subtitles` again and storing the new set.
9. Call `preview_project` and look at the storyboard before rendering. Its
   tiles are in timeline order and cropped the way the render crops, so it
   shows the clip order, the framing, and whether a cut lands on a bad frame,
   for the cost of a few seconds. Fix what is wrong before spending a render.
10. Call `render_project` with `is_preview: true` unless the user asked for the
    final file, and with `burn_subtitles: true` if captions were stored. It
    normalizes the mix to a consistent loudness on its own, so do not try to
    even out clip volumes by hand. Poll `get_job` every few seconds until the
    status is `completed`, `failed`, or `cancelled`. Report the output path
    and length.
11. Render the full-quality version with `is_preview: false` once the user is
    happy, or immediately if they asked for the final file.

For later edits, call `get_project` first so your operations match the current
clips and version.

## Jobs, and picking up where you left off

Projects, assets, analyses, and jobs are kept on disk and survive a restart,
so a conversation can be continued days later.

- When the user refers to work you have no ID for — 「上次那支影片」,
  「我之前匯入的素材」 — call `list_projects` and `list_assets` and ask
  which one they mean. Do not start a new project just because this
  conversation has not seen one; that silently abandons their edit.
- Every `render_project` and `analyze_asset` returns a `job_id`. Hold on to it
  until the job reaches `completed`, `failed`, or `cancelled`.
- If the user changes their mind while a render or an analysis is running, or
  a preview is no longer worth waiting for, call `cancel_job` and tell them
  what you stopped. Do not leave a long job running because they moved on.

## Understanding footage

Use this whenever a request depends on content: what someone says, what is on
screen, where the pauses or mistakes are, or which parts are best.

1. Call `analyze_asset` for each source. Set `language` when you know it. For
   Chinese speech, set `chinese_variant` to match how the user writes:
   `zh-TW` for Traditional Chinese as used in Taiwan (the default for users
   who write Traditional Chinese), `zh-HK` for Hong Kong, `zh-Hant` for
   Traditional without regional wording, or `zh-Hans` for Simplified. Put names or terms the speaker uses into `prompt`. The first
   run downloads the speech model. Poll `get_job`; for long files, tell the
   user the current `stage` and `progress`.
2. Read `get_analysis` in windows of about 10 minutes (`start` / `end`). All
   times are seconds in the source file, so they can be used directly as
   `source_range`.
3. Look before you cut. Call `view_frames` with about 12 frames over the whole
   asset for an overview, then use a narrow range to check a specific moment.
   Do not describe what is on screen without looking. `view_frames` reads a
   source file; `preview_project` shows the edited sequence.
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

## When the user brings raw footage and no plan

Someone who has never edited will hand you a folder and say 「幫我剪一下」.
Do not ask 「你想剪什麼主題？」: not knowing is why they asked. Look at the
footage yourself, then give them a choice between concrete directions.

Work through it in three passes, so the slow one runs only on what survives:

1. **Survey, seconds per file.** `import_folder` on the folder, or
   `import_asset` per file, then `view_frames` with about 6 frames on each
   video. This alone tells you how much footage there is, which files have
   sound, what each one shows, and which are unusable because they are dark,
   shaky, or a stray recording.
2. **Triage, one decoding pass per file.** On the files that look usable,
   call `analyze_asset` with `transcribe: false`. That finds scene changes,
   silences, and black or frozen picture without running speech recognition,
   which is the slow part. Use it to find where anything happens and to drop
   files that are mostly nothing.
3. **Propose, then transcribe.** Offer two or three directions and let the
   user pick. Only once they have picked, run `analyze_asset` with
   transcription, on the files that direction needs.

Never transcribe a whole folder up front. Twenty ten-minute files take hours,
and most of it gets thrown away.

A proposal says what is in the footage, how long each option runs, and what it
leaves out. Read `skill://clip-editing/pacing-and-structure.md` before naming
those lengths. Label the options so the user can answer with one letter:

「這 8 支我都看過了，共 42 分鐘。裡面有三條線：海邊 4 分鐘、晚餐聊天有講到之後的行程 6 分鐘，
另外 2 支畫面晃得很厲害，建議不要用。
A：一天的流水帳，約 90 秒，每個地點各留一點。
B：只留晚餐那段對話，約 2 分鐘，比較完整。
你想要哪一種？」

When the user answers 「你決定」, take the first option, say in one line which
one you took, and show them a storyboard before you render.

## Background music

1. Add an audio track, such as `{"action": "add_track", "track_id": "music",
   "track_type": "audio", "duck_under_speech": true}`, and append the song
   with `insert_clip`. To start the song at a later point, such as the chorus,
   set `source_range.start` to that point. To start the music later in the
   video, use `add_clip` with `timeline_in` instead.
2. Set the music volume; see [Defaults](#defaults). Keep
   `duck_under_speech: true` whenever any video clip has sound, so the music
   gets out of the way of the talking by itself instead of being held quiet
   the whole way through.
3. End the batch with `fit_track` on the music track:
   `{"action": "fit_track", "track_id": "music", "fade_in": 1, "fade_out": 2}`.
   It makes the music cover the video exactly: it trims the clip that runs
   past the end, drops anything entirely past it, plays more of the song when
   the video is longer (and only loops back to the start once the song runs
   out), and puts the fades on whichever clips are first and last afterwards. A fade longer than the clip it lands on is shortened to fit.
   Pass `loop: false` if the user would rather the music stop when the song
   runs out than start over.
   **Never work these lengths out yourself.** There is no arithmetic to do
   here and getting it wrong cuts the music off abruptly.
4. Run `fit_track` again at the end of any later batch that changes the video
   length. It is safe to repeat.

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
| 「把 C 移到最前面」 / move C to the front | `reorder_clip` C with `before_clip_id` = the first clip |
| 「把 A 移到最後」 / move A to the end | `reorder_clip` A without `before_clip_id` |
| 「把這段從中間切開」 / split this clip | `split_clip` with `at` (timeline) or `at_source` (source), plus a `new_clip_id` for the second half |
| 「改成倒敍，結尾放最前面」 / open with the ending | `split_clip` at the boundary, then `reorder_clip` the tail with `before_clip_id` = the first clip |
| 「把 A 換成 d.mp4」 / replace A with d.mp4 | `insert_clip` new clip with `before_clip_id: A`, then `delete_clip` A |
| 「A 和 B 中間停 2 秒黑畫面」 / 2 s of black before B | `move_clip` B and every later video clip, each with `new_timeline_in` set to its current `timeline_in` plus 2 |
| 「加背景音樂 song.mp3」 / add background music | See [Background music](#background-music) |
| 「音樂從第 10 秒才進來」 / start the music at 0:10 | `move_clip` the first music clip with `new_timeline_in: 10` (or `add_clip` with `timeline_in: 10`), then `fit_track` again so the music still ends with the video |
| 「用歌的 1:05 開始」 / start from the chorus at 1:05 | Music clip `source_range.start: 65` |
| 「音樂小聲一點／大聲一點」 / music quieter or louder | `set_clip_audio` on every music clip, `volume` × 0.6 or × 1.5 |
| 「把影片原音關掉」 / mute the original sound | `set_clip_audio` `volume: 0` on every video clip |
| 「音樂淡出」 / fade the music out | `fit_track` with `fade_out: 2`; it trims the music to the video and puts the fade on whichever clip ends up last |
| 「拿掉背景音樂」 / remove the music | `delete_clip` every clip on the audio track |
| 「人聲出現時音樂小聲一點」 / duck the music under the talking | `set_track_audio` on the music track with `duck_under_speech: true` |
| 「每段音量差很多」 / the volume jumps between clips | Nothing: `render_project` normalizes the finished mix to -14 LUFS |
| 「做成直式／方形」 / make it vertical or square | New project with the new size, then rebuild the clips |
| 「開頭淡入、結尾淡出」 / fade in at the start and out at the end | `set_clip_look` `video_fade_in` on the first clip, `video_fade_out` on the last |
| 「兩段之間過一下黑」 / dip through black between two clips | `set_clip_look` `video_fade_out` on the earlier clip and `video_fade_in` on the later one |
| 「這段亮一點／色彩濃一點」 / brighter or more colourful | `set_clip_look` with `color` `brightness` or `saturation`; adjustments you leave out keep their current value |
| 「改成黑白」 / make it black and white | `set_clip_look` with `color` `{"saturation": 0}` |
| 「調色拿掉，回原本的樣子」 / undo the grade on this clip | `set_clip_look` with `clear_color: true` |
| 「小視窗拿掉」 / drop the inset | `delete_clip` it, or `set_clip_look` with `clear_layout: true` to make it cover the frame instead |
| 「色溫暖一點／冷一點」 / warmer or cooler | `set_clip_look` with `color` `temperature`, below 6500 for warmer and above for cooler |
| 「右上角放一個小視窗」 / put an inset in the top right | `add_track` a second video track, then `add_clip` with `timeline_in` and a `layout` box |
| 「中間插一段別的畫面蓋掉原本的」 / cut away to other footage over the same sound | `add_clip` on the upper video track with no `layout`, so it covers the frame, and `volume: 0` so the sound underneath keeps running |
| 「加字幕」 / add captions | `generate_subtitles`, show the user the lines, `set_subtitles`, then render with `burn_subtitles: true` |
| 「字幕有個字打錯了」 / a caption has the wrong word | `get_subtitles` around that moment to find the cue's `id`, then one `edit_subtitle` with its new `text` |
| 「字幕跟畫面對不上了」 / the captions no longer match the picture | The cut changed after they were stored; run `generate_subtitles` again and store the new set with `set_subtitles` |
| 「這句字幕多停一下」 / hold this caption longer | `edit_subtitle` with a new `end` |
| 「這句不要了」 / drop this caption | `edit_subtitle` with `delete: true` |
| 「音樂比影片長／短」 / the music does not match the video length | `fit_track` on the music track |
| 「這支影片在講什麼」 / what is this video about | `analyze_asset`, then summarize `get_analysis` with times, and `view_frames` for the visuals |
| 「剪掉講錯／重講的地方」 / remove flubbed takes | From the transcript, keep only the last complete version of each repeated sentence |
| 「去掉停頓／氣口」 / remove pauses | Keep the speech between `silences` longer than about 0.6 s, with padding, as consecutive clips |
| 「只留有講到 X 的段落」 / keep only parts about X | Keep the transcript segments about X, plus enough context to make sense |
| 「剪掉黑畫面／畫面卡住的地方」 / remove black or frozen parts | Keep the time outside `black_frames` / `frozen_frames` |
| 「找出精華剪成 60 秒」 / a 60 s highlight reel | Choose segments by transcript and frames until about 60 s; confirm the list before rendering |
| 「第 3 分鐘那個畫面是什麼」 / what is on screen at 3:00 | `view_frames` with `start: 175`, `end: 185`, `count: 4` |
| 「現在剪成什麼樣子」 / show me the cut so far | `preview_project`, then describe the order and the cut points |
| 「這個資料夾的影片幫我剪一下」 / edit this folder for me | `import_folder`, then survey and propose; see [When the user brings raw footage and no plan](#when-the-user-brings-raw-footage-and-no-plan) |

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
| Music volume | 0.5 with `duck_under_speech: true` if any video clip has sound, so the music is audible between sentences and drops under them; 0.8 and no ducking if no video clip has sound |
| Output loudness | -14 LUFS (`render_project` does this by default; only pass `loudness_target: null` if the user asks for the raw levels) |
| Colour | As shot. Only adjust when the user asks, or when a clip is clearly too dark next to the others |
| Inset box | About a third of the frame in a corner, clear of the edges, such as `{"x": 0.62, "y": 0.06, "width": 0.32, "height": 0.32}` |
| Captions | Off unless asked. When asked, propose them with `generate_subtitles` and have the user check the wording before rendering |
| Music fades | 1 s fade-in on the first music clip, 2 s fade-out ending at the video end |
| Render | Preview first |
| Length | No default: ask, or propose one. See `skill://clip-editing/pacing-and-structure.md` |

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

When the user has no plan at all, none of this applies: do not ask, survey the
footage and propose directions instead. See
[When the user brings raw footage and no plan](#when-the-user-brings-raw-footage-and-no-plan).

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
| `has no clips on a video track to preview` | Build the video sequence first; a storyboard needs at least one clip. |
| `runs outside the frame` | The inset's `x + width` or `y + height` is over 1.0; shrink it or move it back inside. |
| `has no transcribed clips to caption` | Run `analyze_asset` with transcription on the sources first. |
| `has no captions to burn` | Call `generate_subtitles`, then store them with a `set_subtitles` operation. |
| `caption ... not found` | Read the current `id`s with `get_subtitles`; a caption that was deleted, or a set replaced by freshly generated cues, will not have the old one. |
| `sets the length itself` | `fit_track` is for audio tracks; the video track already decides the length. |
| `the project has no video yet` | Build the video sequence before fitting music to it. |
| `must fall inside` | The split point is outside that clip; check the clip's source range, or split the clip that really covers that moment. |
| `already exists on track` | Choose a `new_clip_id` that is not in use on that track. |
| `not found on track` | Call `get_project` and use the clip IDs it reports. |
| `only audio tracks can duck under speech` | `duck_under_speech` belongs on a music track, not a video one. |
| `is not drawn on top of anything` | A `layout` only works on a video track added above the base one; add that track first. |
| `video fades ... are longer than the clip` | Shorten the fades to fit the clip. |
| `... is not supported yet` | Explain the limit and offer the closest supported result. |
| Job `failed` | Summarize `error_message` for the user. Do not retry the same render unchanged. |

## Further reading

Two parts of this guide live in their own files, because they are only needed
sometimes. Read them with `read_resource` when they apply; `list_resources`
shows everything this server publishes:

- `skill://clip-editing/pacing-and-structure.md` — how long to make the edit,
  what to open on, how fast to cut, and what the frame cuts off. Read it
  before proposing a length or a structure.
- `skill://clip-editing/examples.md` — two requests worked through from the
  user's words to the tool calls and the reply.
