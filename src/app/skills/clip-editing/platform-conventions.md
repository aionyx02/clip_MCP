# Platforms and delivery

Read this when the user names where the video is going — YouTube, Reels,
TikTok, Shorts — or asks for captions in a style, several shapes of one cut,
chapters, a cover, or the edit handed to another editing program, or asks
where a file went or how much space this takes. And read it before the
final render, for the check that comes first. It is referenced
from SKILL.md.

## What each platform expects

| Platform | Shape | Caption preset | Notes |
|---|---|---|---|
| YouTube | landscape, 1920x1080 | `youtube` | Chapters from the parts of the cut; a cover frame |
| YouTube Shorts | portrait, 1080x1920 | `reels` | No preset of its own; the Reels safe area is the closest |
| Instagram Reels | portrait, 1080x1920 | `reels` | The caption, handle and audio credit take the bottom of the frame, so captions sit just above them; word-by-word on |
| TikTok | portrait, 1080x1920 | `tiktok` | As Reels, plus the buttons up the right-hand side push the safe area in further |
| Instagram feed | square, 1080x1080 | `plain` | |

These are the platforms' own layouts, which change when their apps do; the
presets hold the numbers in one place. How long each should run is a matter
of pacing: `skill://clip-editing/pacing-and-structure.md`.

## Caption style

`set_caption_style` decides how burned-in captions are drawn. Start from the
platform the video is going to — `youtube`, `reels`, `tiktok`, `vertical`
(a vertical video minutes long), or `plain` —
because each sets the safe area that platform's own buttons and captions take
up, along with the text size and outline. Anything given alongside the preset
wins, so `{"preset": "tiktok", "style": {"karaoke": false}}` is that safe area
without the word-by-word lighting.

- **One line at a time** (`single_line`, on by default): a caption too long
  for one line is shown as several lines one after another, never stacked up
  into the picture. Set the style before `generate_subtitles`, which then
  keeps each caption to one line of that style. Turn it off only when the user
  asks for stacked captions.
- **Word-by-word lighting** (`karaoke`): each word brightens as it is said. On
  for `reels` and `tiktok`, which is where it is expected. It needs the word
  timings `generate_subtitles` puts on a caption; one written by hand has none
  and simply lights up whole. A correction that swaps characters one for one
  keeps them; one that changes how many there are drops them. For a vertical
  video minutes long, use the `vertical` preset: the same safe area without
  the lighting.
- **Long enough to read** (`min_seconds`, 0.7 by default): a caption on screen
  for less is held on while its shot has room. One that cannot be — the last
  words of a video, with the cut ending right after them — is found by the
  check before a render, with the ways out: let that shot run on, drop the
  caption, or, if the user reads that fast, lower `min_seconds`.
- **Bilingual**: put the second language in a caption's `secondary`, either in
  `edit_subtitle`, or in `add_subtitle` for a new one. It is drawn smaller
  under the first line. Nothing here translates anything — the words are yours.
- **Who is talking** (`speaker_mark`): `name` puts the speaker in front of the
  line, `colour` gives each one their own, `both` does both, `off` says
  nothing. The labels are joined across files, `V1`, `V2`, so one person is
  one label however many cameras recorded them; map them to real names with
  `speaker_names`, for example `{"V1": "阿明"}`. A label with no name keeps the
  label, because `V2` is better than crediting the wrong person. Ask the user
  who is who rather than guessing from the transcript. A file analyzed before
  voices were kept still shows its own `S1`, `S2`, which mean nothing outside
  that file: `analyze_asset` it again (`again: true`) rather than mapping its
  labels to names.

## The check before a render

`render_project` checks the cut first and refuses a full render while it finds
anything; `check_render` runs the same check without rendering, so run it
first. It finds nine kinds of thing: `bad_picture` (black or frozen picture
that reaches the screen), `clipping` (a recording squared off at the ceiling —
turning it down does not undo it), `music` (music too close under somebody
talking — `set_music_level` down, or let it duck), `mid_speech` (a cut inside a word or a
phrase; it names the nearest pause — move the cut there), `repeated` (the same
stretch of a file shown twice), `continuity` (a clip put on by hand between
two shot hours away from it — move it, or let it stand if it is a
flash-forward), `unplanned` (three or more clips put together
by hand, with no plan saying how its parts hold together — ask which structure,
write one and compile it rather than asking to go ahead), `captions` (a caption too tall for the
frame, most often in a portrait render of a landscape cut) and `length` (far
from the length the plan asked for). Each is a fact, not a verdict — the black
may be a deliberate pause, the user may prefer the longer cut — so tell them
in plain words and let them decide. Fix what they want fixed. Only once they
have said to go ahead anyway, render again with those kinds in `allow`; never
fill `allow` in yourself. Previews are never refused.

## One cut, several shapes

`render_project` with `frame` set to `landscape`, `portrait` or `square`, once
per shape. The project is not touched, and the short side stays the same.
Where a shot is a different shape from the frame, the crop follows the face
the analysis found; it holds still, and cuts to a new framing when the face
has stayed near the edge for a while rather than panning after it. Only the
largest face is followed, so in a two-shot interview rendered portrait one of
the two will be out of frame — say so. A shot with no face that a crop would
cut to under 70% — scenery, hands at work, a screen in a vertical video — is
shown whole instead, over a blurred copy of itself. The same holds for a
landscape shot in a vertical project, or the other way round. Each shot can be
told otherwise with `set_clip_look` `fit`: `{"mode": "whole"}` for
「整個畫面都要看到」, `{"mode": "fill"}` to crop it anyway, with `center_x` /
`center_y` (0–1 of the picture) for 「往左一點」 and `zoom` for 「拉近一點」;
`clear_fit` gives the choice back. Look at `preview_project` with the same
`frame` first — the tiles are cropped the way the render will be. Check
captions again for a portrait version: lines that fit across a landscape frame
can stack too tall in a narrow one.

## Chapters and covers

`get_chapters` reads the parts of the cut back as chapters, with the
description text ready to paste. The render carries the same chapters in the
file. YouTube shows chapters only when the first is at 0:00, there are at
least three, and each is at least ten seconds; `problems` says what breaks
that. Fixing it means merging or renaming parts — a change to the plan, so
ask.

`propose_covers` offers one frame per shot — never black, each the moment with
the biggest face or the sharpest picture — as a labeled sheet. Show it, let the
user choose, and save the one they pick with `export_cover` at its time in the
cut.

## Handing over to an editing program

`export_timeline` writes the cut pointed at the original files: `fcpxml` for
Final Cut, Resolve and Premiere, `otio` for Resolve and OpenTimelineIO tools,
`edl` for the sequence alone, and `srt` for the captions. The edit comes
across with its speed changes, on each file's own timecode; `otio` also
carries transitions and `fcpxml` each clip's level. What this server draws
itself does not — colour, fades, voice repair, where an inset sits.
`left_behind` lists which of those this cut uses: say so when you hand over
the file.

## Requests about delivery

| Request | Operations |
|---|---|
| 「同一支也出直式」「Reels 跟 YouTube 各一版」 / one cut for several platforms | `render_project` with `frame: "portrait"` (and again with `landscape` or `square`); `preview_project` with the same `frame` first |
| 「字幕要 TikTok 那種」「一個字一個字跳」 / platform captions, word-by-word | `set_caption_style` with `preset: "tiktok"` (or `reels`, `youtube`, `plain`) |
| 「字幕加英文」「中英對照」 / bilingual captions | Put the other language in each caption's `secondary`, with `set_subtitles` or `edit_subtitle` |
| 「誰在講話要標出來」「兩個人不同顏色」 / show who is speaking | `set_caption_style` with `speaker_mark` and `speaker_names` |
| 「幫我寫 YouTube 章節」 / YouTube chapters | `get_chapters`, paste its `description`; say what `problems` lists |
| 「挑一張封面」「縮圖」 / pick a cover or thumbnail | `propose_covers`, let the user choose, then `export_cover` |
| 「我要拿去 Premiere／達文西／Final Cut 修」 / finish it in another editor | `export_timeline` with `fcpxml` (or `otio`, `edl`), plus `srt` for the captions; read out `left_behind` |

## Where files go, and disk space

A finished video, a cover and a timeline export are the user's: they are
saved in their Videos folder under `clip-mcp`, named after the project
(`EP1 台北.mp4`, `EP1 台北 封面.jpg`), with `(2)` added rather than
overwriting. Tell them that folder when a render finishes. Previews and sound
checks are working files in the workspace; only the newest of each project is
kept, so do not point the user at an old one.

The footage itself is never copied: projects play it from where it is. So
moving or deleting a source file breaks the projects that use it — say so if
the user talks about tidying their footage.

When the user asks how much room this takes, or the disk is full,
`storage_usage` says it in their words. `clean_storage` clears only what can
be made again — `previews`, `thumbnails`, `playback`, `work` — and only the
kinds they agreed to; the playback copies also go by themselves, oldest
first, when the disk runs low. `videos` says what each video's other
versions would give back: deleting them, deleting a project and emptying the
trash are the user's to do in the editor — a project's 版本 tab and 儲存空間 —
so suggest the biggest and say where. A deleted project waits in the trash
for 30 days and can be put back from 儲存空間. An old `outputs` folder from before this layout shows up as
`legacy`: `tidy_old_outputs` without `confirm` lists what would happen,
read that back with the sizes, and confirm only once they agree. Only older
versions of a video, cover or export go, and to the recycle bin rather than
for good; sound checks and render logs stay where they were.
