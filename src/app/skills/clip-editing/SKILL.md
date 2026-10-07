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
| `skill://clip-editing/reading-footage.md` | The request depends on what is said or shown, or the user hands you footage with no plan — and before any `analyze_asset`, since how long it takes, and whether to ask the user first, depends on this computer |
| `skill://clip-editing/editing-intelligence.md` | You are choosing which footage goes in, in what order, what covers it, and what music goes under it — writing a plan |
| `skill://clip-editing/pacing-and-structure.md` | You are about to propose a length, an opening, or how fast to cut |
| `skill://clip-editing/platform-conventions.md` | The user names a platform, asks for captions in a style, several shapes, chapters, a cover, or an export, asks where a file went or how much space this takes — and before any final render |
| `skill://clip-editing/feedback-handling.md` | The user has watched a version and says what is wrong, wants the last one back, or asks what changed |
| `skill://clip-editing/request-map.md` | A request does not plainly say which operation it is — 「把第二段換到最後」, 「成片第 12 秒那裡」 — or you need how times in the cut map to the source |
| `skill://clip-editing/examples.md` | You want to see a whole exchange from the user's words to the reply |

## What is supported today

- Cutting, joining, trimming, splitting, reordering, inserting and replacing
  clips on one sequence, including non-linear orders; black gaps; a plan the
  server compiles into the cut, cleaning it up as it goes.
- Mixed sources — any resolution, orientation or frame rate, with or without
  sound — scaled to fill the frame and cropped around the face in them. One
  cut rendered in several shapes.
- Music on audio tracks, ducking under speech, changing with the parts of
  the video, and cuts put on its beat. Where the footage is transcribed, the
  music drops wherever somebody is talking, however quietly. Every render
  normalized to -14 LUFS.
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
  returns `duration`, the length of the edited video. A project must have a
  `name` — the editor app, `list_projects` and the finished file all show
  it — so name it for what the video is; `rename_project` changes it later.
  An older project with none: propose a name and rename it.
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
  go with it, drop it and they stop appearing. `generate_subtitles` makes
  them from the transcripts of what is heard in the cut — not insets, not
  music, not a clip at `volume: 0` — stores them, and returns one line per
  caption to proofread. Captions already corrected for the same footage, in
  any project, are used instead of the transcript, and the workspace glossary
  (`fix_words`) corrects words the transcriber always gets wrong. They only
  appear in the file with `burn_subtitles`; `set_caption_style` decides how they are drawn, from
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
- **Changing a clip in place**: `set_clip_audio` (volume, fades, J and L
  cuts, voice repair), `set_clip_look` (transition, colour, box),
  `set_clip_speed`, and `set_clip_pinned` to keep or hand back a clip
  adjusted by hand; `set_track_audio` for a whole track's ducking, and
  `set_markers` for where each part begins. The request-map file says which
  request takes which.

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

**When the user says to finish on your own** — 「直接做完再給我看」 — skip the
read-back, not the checks: render a preview, look at it with `view_render`
(`at: "captions"`, then `"transitions"`), hear any join in doubt with
`listen_again` and the preview's `job_id`, fix what you find, then render
the final cut. `allow` is still never yours to fill in: when `check_render`
finds something you cannot fix, stop and ask. Then report every decision you
made — what went in, what was left out and why, the music — since the user
was not there to agree to them.

## Workflow

1. Read the request and settle anything missing; see
   [When the request is incomplete](#when-the-request-is-incomplete). When the
   user brings unedited footage and no plan, read the reading-footage file.
2. Call `import_asset` for every source file, including music, or
   `import_folder` when the user points at a folder, with `recursive: true`
   when the footage sits in sub-folders. Use the returned `duration` instead
   of guessing lengths; `inspect_media` gives resolution and frame rate.
3. When the edit depends on what is said or shown, read the footage first.
   Whenever it uses more than one file or has more than one part, write a
   plan in the structure the user chose — 起承轉合, 清單式, 教學步驟 and the
   rest are in the editing-intelligence file — and compile it
   (`save_plan`, `validate_plan`, `compile_plan`) rather than placing clips
   by hand; the check before a render finds a hand-built sequence. A note
   with `look: true` is worth weighing, and worth telling the user about;
   its `clip_ids` say which pieces, its `message` what to do.
4. By hand: `create_project` with the output settings and a `name` the user
   would recognise, such as `EP1 台北`; see [Defaults](#defaults). Then
   `apply_edits` with an `add_track` operation,
   `{"action": "add_track", "track_id": "main", "track_type": "video"}`,
   together with the first clips, built with `insert_clip` — without
   `before_clip_id`, each is appended after the last. Give clips short IDs
   such as `intro` or `c1`. Put all operations for one request into a single
   `apply_edits` call.
5. Add music if requested; see [Background music](#background-music).
6. If the user asked for captions, set the caption style first (the platform
   file), because each caption is proposed one line wide for that style. Then
   call `generate_subtitles` and read every line it returns — transcripts
   mishear names — starting with its `to_check`, the lines nobody has read
   yet, unsure ones first; `get_subtitles` says where each came from and
   marks the unsure words. Correct a line with `edit_subtitle` (`cue_id`; times in the
   cut with `timeline_start`/`timeline_end`), add one with `add_subtitle`, and
   read back what the edit returns. A correction of what was said serves every
   project; words for this cut only take `this_project_only`.
   `set_subtitles` replaces every caption at once, for a set written
   elsewhere — a translation, a script. Lower
   `max_characters` or `max_seconds` for shorter lines on screen.
7. Call `preview_project` and look at the storyboard before rendering: the
   clip order, the framing, a cut on a bad frame, for a few seconds' cost. If
   there is music, call `preview_sound` too — you cannot hear it, and the
   storyboard cannot show whether it gets out of the way of the talking.
8. Call `check_render` and tell the user what it finds; a full render is
   refused until they have heard. See the platform file.
9. Call `render_project` with `is_preview: true` unless the user asked for
   the final file, and with `burn_subtitles: true` if captions were stored.
   Do not even out clip volumes by hand: the render normalizes the mix. Call
   `get_job` with `wait_seconds: 50` until the status is `completed`, `failed`,
   or `cancelled`. Look at what came out with `view_render` — captions and
   transitions only exist once rendered — before reporting the path and length.
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
- Every `render_project` and `analyze_asset` returns job IDs. Hold on to them
  until each reaches `completed`, `failed`, or `cancelled`; `get_job` takes
  them all at once, and with `wait_seconds` waits for one to end.
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

### Where the music comes from

The user's songs are in the library's music half. When they want music and
have not named a song, `find_music` with what they asked for in English and
the length of the part as `min_seconds`; read the top two or three back with
how each sounds (「聽起來偏懷舊、原聲吉他」 — the mood is a model's ear) and
let them choose. When reading a plan back, say the mood of the song in it the
same way. A compile's `music_loops`, `music_peak_off` and `music_cut_off`
notes say where a song sits badly and how to fix it: tell the user, and fix
it only if they want. When nothing in the library fits, ask for a file, and
in the same message tell them in plain words what each kind of source means
for where the video is going:

- **A commercial song** — bought, or from a streaming app: fine for a video
  only they watch. Posted to YouTube, Instagram or TikTok, it is almost
  always caught by the platform's copyright matching: muted, blocked in some
  countries, or its ad money goes to the song's owner.
- **The app's own music**: a song added inside Instagram Reels or TikTok is
  licensed by the platform, and is the safest choice there. Render without
  music and let them add it in the app; say that captions and cuts are
  already in place.
- **Royalty-free libraries** such as the YouTube Studio audio library or
  Pixabay Music: free to use in videos. Some tracks ask for a credit line in
  the description, which the download page says.
- **Creative Commons music**: read the licence. `NC` means no commercial use
  (a sponsored post, a client's video), `ND` means no changes — trimming and
  looping may already be one.
- **A subscription library** such as Epidemic Sound or Artlist: covered for
  videos published while subscribed, usually tied to their channel.

Say once that this is general guidance, not legal advice, and that the
licence on the page they downloaded from is what counts. Suggest they keep
where each song came from for the credit line.

When this client can search the web, you may suggest a few tracks from those
libraries that fit the structure and the mood — a song with a steady beat for
a 蒙太奇, something that lifts at the turn for 起承轉合 — with links. Do not
download one yourself: the licence is something the user agrees to, and
which terms are enough depends on whether the video is commercial, which only
they know. They download it and hand you the file.

## Defaults

Use these without asking, and mention the ones you chose in one short line.

| Setting | Default |
|---|---|
| Output size | 1080x1920 for 直式, 短影音, Reels, Shorts, 抖音, or TikTok; 1080x1080 for 方形 or IG 貼文; 1920x1080 otherwise, or 1080x1920 if most sources are vertical |
| Frame rate | 30 (`fps_num: 30, fps_den: 1`). Use 25, 60, or 29.97 (`30000/1001`) only if asked |
| Clip order | The order the user listed files in; otherwise file-name order |
| Music volume | 0.5 with `duck_under_speech: true` if any video clip has sound; 0.8 and no ducking if none has. Where the footage is transcribed, a music clip's volume is measured from the talking, not from how loud the song was mastered: 0.5 sits 6 dB under the voices between sentences and 16 dB under while somebody talks |
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
  there is no built-in music library, so ask for a music file, and say what
  each kind of source means for where the video is going
  ([Where the music comes from](#where-the-music-comes-from)).
- The structure of anything more than a trim, unless the user named one:
  offer the two or three that suit the footage (the editing-intelligence
  file) and let them pick. It decides what goes first and what it ends on,
  so a wrong guess is a whole recut.
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
| `has no captions to burn` | Call `generate_subtitles`. |
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
