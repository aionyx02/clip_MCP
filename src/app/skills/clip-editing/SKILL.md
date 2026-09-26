---
name: clip-editing
description: Turn natural-language video editing requests (for example 「把這三支影片接起來」, 「剪掉講錯的地方」, 「加背景音樂」, "move the last clip to the front") into clip-mcp tool calls. Use it whenever the user wants to understand, cut, join, trim, reorder, add music to, or render videos with this server, including when the request depends on what is said or shown in the footage, when the user points at a whole folder of unedited footage without saying what to make of it, or when the request is vague or missing details.
---

# Clip editing

This server edits videos on a timeline and renders MP4 files with FFmpeg.
This file says which tool a request turns into. Reply to the user in their
language.

## When to read the other files

Each of these answers one question. Read it with `read_resource` when that
question comes up, not before; `list_resources` shows everything published.

| Read | When |
|---|---|
| `skill://clip-editing/reading-footage.md` | The request depends on what is said or shown, or the user hands you footage with no plan |
| `skill://clip-editing/editing-intelligence.md` | You are choosing which footage goes in, in what order, what covers it, and what music goes under it — writing a plan |
| `skill://clip-editing/pacing-and-structure.md` | You are about to propose a length, an opening, or how fast to cut |
| `skill://clip-editing/platform-conventions.md` | The user names a platform, asks for captions in a style, several shapes, chapters, a cover, or an export — and before any final render |
| `skill://clip-editing/feedback-handling.md` | The user has watched a version and says what is wrong, wants the last one back, or asks what changed |
| `skill://clip-editing/examples.md` | You want to see a whole exchange from the user's words to the reply |

## What is supported today

- Cutting, joining, trimming, splitting, reordering, inserting and replacing
  clips on one sequence, including non-linear orders; black gaps; a plan the
  server compiles into the cut, cleaning it up as it goes.
- Mixed sources — any resolution, orientation or frame rate, with or without
  sound — scaled to fill the frame and cropped around the face in them. One
  cut rendered in several shapes.
- Music on audio tracks, ducking under speech, changing with the parts of
  the video, and cuts put on its beat. Every render normalized to -14 LUFS.
- Fades, transitions (dissolve, wipe, dip through a colour), colour per clip,
  speed changes, J and L cuts, picture in picture, covering picture (B-roll).
- Voice repair and matching two people's levels.
- Burned-in captions from the transcript, in platform styles, word by word,
  bilingual, with who is talking.
- Reading footage: transcripts, scenes, silences, bad picture, faces and
  speakers, as searchable clips; frame contact sheets.
- Storyboards in seconds, previews that only redo what changed, a sound
  preview, a picture of what a change did, and a version history of every
  plan.
- A check before rendering, chapters, covers, and export for Premiere,
  Resolve and Final Cut.

Not supported yet: still images, free text and graphics other than captions,
and filters beyond the colour controls. When a request needs one of these,
say so plainly, offer the closest supported result, and never pretend it was
done. For example:
「目前還放不進靜態圖片。這一段我可以用附近的空鏡蓋過去，可以嗎？」

Transitions end on the cut rather than straddling it, so adding one never
changes how long the video runs or moves anything after it. A transition needs
a clip ending exactly where the one taking it begins; to come up from black at
the very start of a video, use `video_fade_in` instead.

## Concepts

- **Asset**: a media file registered with `import_asset`. Its `duration` is
  the source length in seconds. `has_video` and `has_audio` tell you which
  kind of track it can go on.
- **Project**: output `width`, `height`, and frame rate, plus its tracks.
  These settings cannot be changed later; create a new project instead, or
  render the same one in another shape with `frame`. `get_project` also
  returns `duration`, the length of the edited video. Give a project a `name`,
  because `list_projects` shows it and three projects cannot be told apart by
  their IDs; `rename_project` changes it later.
- **Tracks**: the first `video` track is the base: it holds the sequence and
  each clip's own sound, and it decides the output length. Further `video`
  tracks are drawn on top of it; their clips carry their own sound too. Any
  number of `audio` tracks hold music. Anything past the end of the base
  track is cut off.
- **Layout**: a clip on a video track above the base is drawn in the box its
  `layout` gives, as fractions of the frame: `{"x": 0.62, "y": 0.06,
  "width": 0.32, "height": 0.32}` is a small inset in the top right. A clip
  with no layout covers the whole frame, which is how you cut away to another
  shot. A `layout` on a base clip is refused rather than ignored.
- **Clip**: the part of an asset between `source_range.start` and
  `source_range.end` (seconds in the source file), placed on a track at
  `timeline_in`. Its length on the timeline is `end - start` at normal speed.
  It also has `volume` (1.0 is the original level) and `audio_fade_in` /
  `audio_fade_out` in seconds.
- **Captions**: anchored to the file and the second the words were spoken, not
  to a moment in the cut, so they follow the footage: move a clip and its lines
  go with it, drop it and they stop appearing. `generate_subtitles` proposes
  them from the transcripts of what is heard in the cut — not insets, not
  music, not a clip at `volume: 0` — and counts how many land `overlapping`.
  Nothing is saved until `set_subtitles`, and they only appear in the file
  with `burn_subtitles`; `set_caption_style` decides how they are drawn, from
  a platform `preset` (the platform file). `get_subtitles` reports `total`, how many land in the
  cut, and `stored`; `stored` above `total` means some belong to footage the
  edit dropped.
- **Version**: `apply_edits` needs the project's current `version` as
  `expected_version`. `get_project` leaves out clip fields still at their
  default, so a clip with no `volume` is at 1.0 and one with no `color` is as
  shot; it reports captions only as `subtitle_count` — read them with
  `get_subtitles`, or pass `include_subtitles: true`. A failed call changes
  nothing, so fix the operations and retry with the same version.
- **Magnetic editing**: `insert_clip` pushes later clips on the same track
  back to make room. `trim_clip` and `delete_clip` pull later clips forward to
  close gaps, unless `ripple` is false. `reorder_clip` moves a clip to another
  place in one step and keeps its cut, volume and fades — use it instead of
  deleting and re-inserting. `split_clip` cuts a clip in two by `at` on the
  timeline or `at_source` in the source, and moves nothing; both halves keep
  the colour and box, and the fades stay on the outer edges. Only `add_clip`
  and `move_clip` use absolute positions, and they never move other clips.
  Clips on other tracks never move, so music stays put when you edit the
  video. Prefer the magnetic operations so you never calculate positions.

## Talking to the user

The person using this server is editing their own video, not writing code.
Write to them as an editor would, not as a program reporting its state.

- Lead with the result: what was made, how long it is, and the file path.
  Then one short line on what you decided for them. Nothing else.
- Keep replies to three or four lines. Say what changed, not how.
- Never put tool names, asset or clip IDs, JSON, or version numbers in a
  reply, and translate errors instead of quoting them: not
  「apply_edits version conflict」, but 「我對了一下目前的順序，已經接好了」.
- Write times as 0:12 or 1:05, and lengths in plain words such as 「33 秒」.
- Give two or three concrete options to choose from instead of an open
  question such as 「你想怎麼剪？」.

**Read the plan back before anything expensive.** A render or an
`analyze_asset` costs minutes, while `preview_project` takes seconds, so
before the first `render_project` of a request, and before analyzing a file
longer than about ten minutes, restate the plan in one sentence in plain
language and wait for a yes. If it involves a length or a structure you
chose, read the pacing file first, so the sentence is one worth agreeing to.
When the edit is already built, send the storyboard with it:

「我理解成：trip1.mp4 和 trip2.mp4 接起來、trip1 只要 5 到 20 秒，做成直式。先出預覽？」

Do this once per request, not once per operation, and skip it when the user
has already approved this exact plan. State your assumptions in it, and the
user corrects what is wrong before a render is wasted.

## Workflow

1. Read the request and settle anything missing; see
   [When the request is incomplete](#when-the-request-is-incomplete). When the
   user brings unedited footage and no plan, read the reading-footage file.
2. Call `import_asset` for every source file, including music, or
   `import_folder` when the user points at a folder, with `recursive: true`
   when the footage sits in sub-folders. Use the returned `duration` instead
   of guessing lengths; `inspect_media` gives resolution and frame rate.
3. When the edit depends on what is said or shown, read the footage first,
   and for anything longer than a handful of clips write a plan and compile
   it (`save_plan`, `validate_plan`, `compile_plan`) rather than placing clips
   by hand.
4. By hand: `create_project` with the output settings and a `name` the user
   would recognise, such as `EP1 台北`; see [Defaults](#defaults). Then
   `apply_edits` with an `add_track` operation,
   `{"action": "add_track", "track_id": "main", "track_type": "video"}`,
   together with the first clips, built with `insert_clip` — without
   `before_clip_id`, each is appended after the last. Give clips short IDs
   such as `intro` or `c1`. Put all operations for one request into a single
   `apply_edits` call.
5. Add music if requested; see [Background music](#background-music).
6. If the user asked for captions, call `generate_subtitles`, show them the
   lines — transcripts mishear names — and store them with `set_subtitles`
   once they are happy. Correct single lines afterwards with `edit_subtitle`
   and its `cue_id`; never resend the whole set to change one word. Lower
   `max_characters` or `max_seconds` for shorter lines on screen.
7. Call `preview_project` and look at the storyboard before rendering: the
   clip order, the framing, a cut on a bad frame, for a few seconds' cost. If
   there is music, call `preview_sound` too — you cannot hear it, and the
   storyboard cannot show whether it gets out of the way of the talking.
8. Call `check_render` and tell the user what it finds; a full render is
   refused until they have heard. See the platform file.
9. Call `render_project` with `is_preview: true` unless the user asked for
   the final file, and with `burn_subtitles: true` if captions were stored.
   Do not even out clip volumes by hand: the render normalizes the mix. Poll
   `get_job` every few seconds until the status is `completed`, `failed`, or
   `cancelled`. Report the output path and length.
10. Render with `is_preview: false` once the user is happy, or at once if they
    asked for the final file.

For later edits, call `get_project` first so your operations match the current
clips and version.

## Jobs, and picking up where you left off

Projects, assets, analyses, plans and jobs are kept on disk and survive a
restart, so a conversation can be continued days later.

- When the user refers to work you have no ID for — 「上次那支影片」 — call
  `list_projects` and `list_assets` and ask which one they mean. Do not start
  a new project just because this conversation has not seen one.
- Every `render_project` and `analyze_asset` returns a `job_id`. Hold on to it
  until the job reaches `completed`, `failed`, or `cancelled`.
- Jobs queue: only one or two run at a time. A job at `queued` with a `stage`
  such as `waiting for 1 running job(s) to finish` is working as intended —
  keep polling `get_job`, and tell the user how many are in line rather than
  that something is stuck.
- If the user changes their mind while a job is running, call `cancel_job`
  and tell them what you stopped.

## Background music

In a plan, music is part of the plan; see the editing-intelligence file. By
hand:

1. Add an audio track, such as `{"action": "add_track", "track_id": "music",
   "track_type": "audio", "duck_under_speech": true}`, and append the song
   with `insert_clip`. To start at a later point in the song, set
   `source_range.start`; to start the music later in the video, use
   `add_clip` with `timeline_in`.
2. Keep `duck_under_speech: true` whenever any video clip has sound, so the
   music gets out of the way of the talking by itself.
3. End the batch with `fit_track`:
   `{"action": "fit_track", "track_id": "music", "fade_in": 1, "fade_out": 2}`.
   It makes the music cover the video exactly — trims, extends, loops, and
   puts the fades on whichever clips end up first and last. Pass
   `loop: false` if the music should stop when the song runs out. **Never
   work these lengths out yourself**, and run it again after any batch that
   changes the video length.

## Mapping requests to operations

Times like `1:30` mean 90 seconds. Unless the user refers to the edited result,
times refer to the source file. Requests about content, the plan, delivery and
feedback are mapped in their own files.

| Request | Operations |
|---|---|
| 「把 a.mp4、b.mp4 接起來」 / join these videos | `insert_clip` for each file in order, `source_range` `{start: 0, end: duration}` |
| 「a.mp4 只要 10 到 20 秒」 / use 0:10–0:20 of a.mp4 | `insert_clip` with `source_range` `{start: 10, end: 20}` |
| 「剪掉 X 開頭 3 秒」 / cut the first 3 s of X | `trim_clip` X, `new_source_range` `{start: start + 3, end: end}` |
| 「剪掉 X 最後 3 秒」 / cut the last 3 s of X | `trim_clip` X, `new_source_range` `{start: start, end: end - 3}` |
| 「刪掉第二段」 / delete the second clip | `delete_clip` (later clips move up); with `ripple: false` to keep the gap |
| 「在 A 和 B 中間插入 c.mp4」 / insert between A and B | `insert_clip` with `before_clip_id: B` |
| 「把 C 移到最前面」 / move C to the front | `reorder_clip` C with `before_clip_id` = the first clip; without it, to the end |
| 「把這段從中間切開」 / split this clip | `split_clip` with `at` (timeline) or `at_source` (source), plus a `new_clip_id` for the second half |
| 「改成倒敍，結尾放最前面」 / open with the ending | `split_clip` at the boundary, then `reorder_clip` the tail with `before_clip_id` = the first clip |
| 「把 A 換成 d.mp4」 / replace A with d.mp4 | `insert_clip` new clip with `before_clip_id: A`, then `delete_clip` A |
| 「A 和 B 中間停 2 秒黑畫面」 / 2 s of black before B | `move_clip` B and every later video clip, each with `new_timeline_in` set to its current `timeline_in` plus 2 |
| 「音樂從第 10 秒才進來」 / start the music at 0:10 | `move_clip` the first music clip with `new_timeline_in: 10`, then `fit_track` again |
| 「用歌的 1:05 開始」 / start from the chorus at 1:05 | Music clip `source_range.start: 65` |
| 「音樂小聲一點／大聲一點」 / music quieter or louder | `set_clip_audio` on every music clip, `volume` × 0.6 or × 1.5 (in a plan: `set_music_level`) |
| 「音樂淡出」「音樂比影片長／短」 / fade the music, fit it | `fit_track` with `fade_out: 2` |
| 「拿掉背景音樂」 / remove the music | `delete_clip` every clip on the audio track |
| 「人聲出現時音樂小聲一點」 / duck the music under the talking | `set_track_audio` on the music track with `duck_under_speech: true` |
| 「把影片原音關掉」 / mute the original sound | `set_clip_audio` `volume: 0` on every video clip |
| 「聲音先進來」「上一句講完再切」 / J cut, L cut | `set_clip_audio` with `audio_lead` on the incoming clip, or `audio_lag` on the outgoing one |
| 「每段音量差很多」 / the volume jumps between clips | Nothing: `render_project` normalizes the finished mix |
| 「這裡用溶接」「不要硬切」 / cross dissolve | `set_clip_look` on the incoming clip with `transition_in: {kind: "dissolve", seconds: 1}` |
| 「用擦劃轉場」「從左邊掃過去」 / wipe | `set_clip_look` with `transition_in: {kind: "wipe", seconds: 0.6, direction: "left"}` |
| 「這裡淡到黑再進來」「過白場」 / dip through a colour | `set_clip_look` with `transition_in: {kind: "dip", seconds: 1, through: "black"}`; `white` or a hex colour also work |
| 「轉場拿掉，改回硬切」 / back to a straight cut | `set_clip_look` with `clear_transition: true` |
| 「開頭淡入、結尾淡出」 / fade in and out | `set_clip_look` `video_fade_in` on the first clip, `video_fade_out` on the last |
| 「這段快轉」「放慢一點」 / speed it up or slow it down | `set_clip_speed` with `speed`; 2.0 is twice as fast. Voices keep their pitch unless `preserve_pitch: false` |
| 「這段亮一點／色彩濃一點」 / brighter or more colourful | `set_clip_look` with `color` `brightness` or `saturation`; ones you leave out keep their value |
| 「改成黑白」 / black and white | `set_clip_look` with `color` `{"saturation": 0}` |
| 「色溫暖一點／冷一點」 / warmer or cooler | `set_clip_look` with `color` `temperature`, below 6500 for warmer |
| 「調色拿掉」 / undo the grade | `set_clip_look` with `clear_color: true` |
| 「右上角放一個小視窗」 / an inset in the top right | `add_track` a second video track, then `add_clip` with `timeline_in` and a `layout` box |
| 「小視窗拿掉」 / drop the inset | `delete_clip` it, or `set_clip_look` with `clear_layout: true` to cover the frame |
| 「中間插一段別的畫面蓋掉原本的」 / cut away over the same sound | `add_clip` on the upper video track with no `layout` and `volume: 0` (in a plan: `add_broll`) |
| 「這段我自己調的不要動」 / keep my version | Already kept: a hand edit pinned it. `set_clip_pinned` `pinned: true` for an untouched clip |
| 「標一下開場到哪裡」 / mark where a part begins | `set_markers`; `compile_plan` already writes one per beat |
| 「這支叫 EP1 台北」 / name this project | `rename_project` with the new `name` |
| 「做成直式／方形」 / vertical or square | `render_project` with `frame`; see the platform file |
| 「加字幕」 / add captions | `generate_subtitles`, show the user the lines, `set_subtitles`, then render with `burn_subtitles: true` |
| 「字幕有個字打錯了」 / a caption has the wrong word | `get_subtitles` around that moment for its `cue_id`, then `edit_subtitle` with the new `text` |
| 「後來又加了一段，那段沒有字幕」 / new footage has no captions | `generate_subtitles` again and store the new set |
| 「這句字幕多停一下」 / hold this caption longer | `edit_subtitle` with a new `source_end` |
| 「這句不要了」 / drop this caption | `edit_subtitle` with `delete: true` |
| 「現在剪成什麼樣子」 / show me the cut so far | `preview_project`, then describe the order and the cut points |
| 「先聽聽看」 / let me hear it | `preview_sound`, and give the user the mix file |

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
| Music volume | 0.5 with `duck_under_speech: true` if any video clip has sound; 0.8 and no ducking if none has |
| Music fades | 1 s fade-in on the first music clip, 2 s fade-out ending at the video end |
| Output loudness | -14 LUFS; only pass `loudness_target: null` if the user asks for the raw levels |
| Colour | As shot, unless asked, or a clip is clearly too dark next to the others |
| Inset box | About a third of the frame in a corner, such as `{"x": 0.62, "y": 0.06, "width": 0.32, "height": 0.32}` |
| Captions | Off unless asked; when asked, have the user check the wording before rendering |
| Render | Preview first |
| Length | No default: ask, or propose one from the pacing file |

## When the request is incomplete

Ask only when a wrong guess would waste a render or go against what the user
wants:
- No source file is named, or several could match. This includes 「加點音樂」:
  there is no built-in music library, so ask for a music file.
- It is unclear which part to keep or remove, as in 「剪短一點」. Propose
  concrete choices: 「要保留前 30 秒，還是去掉開頭和結尾各 5 秒？」
- The order of several clips is unclear and cannot be inferred.

When the user has no plan at all, do not ask: survey the footage and propose
directions instead (the reading-footage file). Do not ask about anything you
can look up — durations, the current timeline, what is said or shown — or
about [Defaults](#defaults). Put all open questions into one short message
with options the user can pick from.

## Handling errors

| Error | What to do |
|---|---|
| `version conflict` | Call `get_project` (or `get_plan`), re-plan against the latest, and retry. |
| `overlaps clip` | Use `insert_clip` or rippling operations instead of absolute positions. |
| `source range ends at ... but asset ... is only ...` | Clamp `end` to the asset duration and tell the user. |
| `has no audio stream, so it cannot be used on audio track` | That file has no sound; ask for another music file. |
| `put audio-only assets on an audio track` | Move the file to an audio track. |
| `audio fades ... are longer than the clip` / `video fades ...` | Shorten the fades to fit the clip. |
| `shorter than one frame` | Drop the clip or make it longer. |
| `no clips on a video track to render` | Add video clips first; the server cannot render music alone. |
| `has not been analyzed; call analyze_asset first` | Run `analyze_asset`, wait for the job, then retry. |
| `has no video frames to show` | The asset is audio-only; use `get_analysis` instead. |
| `has no clips on a video track to preview` | Build the video sequence first. |
| `runs outside the frame` | The inset's `x + width` or `y + height` is over 1.0; shrink it or move it back. |
| `has no transcribed clips to caption` | Run `analyze_asset` with transcription on the sources first. |
| `has no captions to burn` | Call `generate_subtitles`, then store them with `set_subtitles`. |
| `caption ... not found` | Read the current `cue_id`s with `get_subtitles`. |
| `name assets that are not imported` | A caption names footage that is not registered; import it, or drop the caption. |
| `none of the footage they transcribe is in the cut` | Run `generate_subtitles` again against the sequence as it stands. |
| `sets the length itself` | `fit_track` is for audio tracks. |
| `the project has no video yet` | Build the video sequence before fitting music to it. |
| `must fall inside` | The split point is outside that clip; split the clip that really covers that moment. |
| `already exists on track` | Choose a `new_clip_id` that is not in use on that track. |
| `not found on track` | Call `get_project` and use the clip IDs it reports. |
| `only audio tracks can duck under speech` | `duck_under_speech` belongs on a music track. |
| `is not drawn on top of anything` | A `layout` only works on a video track above the base one. |
| `this plan cannot be compiled yet` | Fix each listed problem in the plan; nothing was compiled. |
| `compiling would undo work already on this project` | A hand-adjusted clip's footage left the plan: ask the user whether to put it back or unpin it. |
| `the cut is not ready to render` | Tell the user each finding in plain words and ask; render with `allow` only for the kinds they chose to keep. |
| `... is not supported yet` | Explain the limit and offer the closest supported result. |
| Job `failed` | Summarize `error_message` for the user. Do not retry the same render unchanged. |
