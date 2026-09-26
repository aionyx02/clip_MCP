# Platforms and delivery

Read this when the user names where the video is going — YouTube, Reels,
TikTok, Shorts — or asks for captions in a style, several shapes of one cut,
chapters, a cover, or the edit handed to another editing program. And read
it before the final render, for the check that comes first. It is referenced
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
platform the video is going to — `youtube`, `reels`, `tiktok`, or `plain` —
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
  and simply lights up whole. Correcting a caption's text drops its timings,
  so run `generate_subtitles` again if you want them back.
- **Bilingual**: put the second language in a caption's `secondary`, either in
  the `set_subtitles` you store or with `edit_subtitle`. It is drawn smaller
  under the first line. Nothing here translates anything — the words are yours.
- **Who is talking** (`speaker_mark`): `name` puts the speaker in front of the
  line, `colour` gives each one their own, `both` does both, `off` says
  nothing. The labels are joined across files, `V1`, `V2`, so one person is
  one label however many cameras recorded them; map them to real names with
  `speaker_names`, for example `{"V1": "阿明"}`. A label with no name keeps the
  label, because `V2` is better than crediting the wrong person. Ask the user
  who is who rather than guessing from the transcript. A file analyzed before
  voices were kept still shows its own `S1`, `S2`, which mean nothing outside
  that file: `analyze_asset` it again rather than mapping its labels to names.

## The check before a render

`render_project` checks the cut first and refuses a full render while it finds
anything; `check_render` runs the same check without rendering, so run it
first. It finds seven kinds of thing: `bad_picture` (black or frozen picture
that reaches the screen), `clipping` (a recording squared off at the ceiling —
turning it down does not undo it), `mid_speech` (a cut inside a word or a
phrase; it names the nearest pause — move the cut there), `repeated` (the same
stretch of a file shown twice), `unplanned` (three or more clips put together
by hand, with no plan saying how the video opens, turns and ends — write one
and compile it rather than asking to go ahead), `captions` (a caption too tall for the
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
the two will be out of frame — say so. Look at `preview_project` with the same
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
