# Reading footage

Read this when a request depends on what is said or shown — what someone
says, what is on screen, where the pauses or mistakes are, which parts are
best — and when the user hands you a folder of unedited footage without
saying what to make of it. It is referenced from SKILL.md.

## Understanding footage

1. Call `analyze_asset` for each source. Set `language` when you know it.
   Set `speakers` when you know how many people are talking — told the
   number, the server cannot split one person into two. For Chinese speech,
   set `chinese_variant` to match how the user writes: `zh-TW` for
   Traditional Chinese as used in Taiwan (the default for users who write
   Traditional Chinese), `zh-HK` for Hong Kong, `zh-Hant` for Traditional
   without regional wording, or `zh-Hans` for Simplified. Put names or terms
   the speaker uses into `prompt`. The first run downloads the speech model.
   Poll `get_job`; for long files, tell the user the current `stage` and
   `progress`. A music file — sound with no picture — also gets its beat
   measured, which cutting on the beat needs.
2. Call `build_semantic_timeline` once over the analyzed sources. It turns
   the transcripts and detections into clips you can search — one per
   sentence, per shot, per long pause — each with a stable ID. Keep the
   `timeline_id`. Building again over unchanged analyses is free and returns
   the same IDs, so call it rather than trying to remember one.
3. Find material with `query_clips`, not by reading a transcript end to end.
   Ask for what you need and let the search narrow it: a kind, a phrase, a
   minimum length, a stretch of the file, a bound on the scores. The scores
   cover how well a shot was shot as well as what is in it — exposure, blur,
   camera shake, whether anybody is on screen and where, and the sound's
   level, noise floor and clipping. They are measurements, not verdicts:
   there is no number above which a shot is too dark or too wobbly, so read
   a few and set the bound from what this footage actually looks like. On a
   conversation each clip also carries a `speaker` label, joined across the
   files of the timeline: `V1` is the same person in every file they appear
   in, even where each recording called them something different. Then call
   `get_semantic_clip` on the few worth a closer look, which gives the full
   text, every score, and with `include_words` every word's timing.
   `get_analysis` is still there for checking what a detector actually found,
   but it is not how to read footage any more.
4. On anything longer than a few minutes, break it into sections before you
   plan. Sentences are too fine to think in; sections are one per thing the
   speaker gets through. Call `propose_sections`, read the utterances against
   the candidate boundaries it offers, pick the ones that are real, name each
   section, and send them back with `set_sections`. You can only choose among
   the candidates, which is why a section can never begin mid-word. Give
   sections that are about the same thing the same `topic` — topics are
   labels, so they gather sections across different files, and they are what
   answers 「這批素材有什麼」.
5. Look before you cut. Call `view_frames` with about 12 frames over the whole
   asset for an overview, then use a narrow range to check a specific moment.
   It takes a list of assets, so a range is only accepted for one of them; for
   several, it takes that many frames from each and puts them on one sheet, up
   to 36 frames in total. Do not describe what is on screen without looking.
   `view_frames` reads source files; `preview_project` shows the edited
   sequence.
6. To make what is on screen searchable, call `frames_for_clips` on the clips
   you care about, then write what you saw back with `set_clip_tags`. Show the
   user your descriptions first and let them correct anything you misread:
   these are stored as facts about the footage and later choices are made from
   them. What wrote each one is stored alongside it, so a label you read off a
   thumbnail is never mistaken later for something a detector measured.
7. Turn clips into edits. Whenever the edit uses more than one file or has
   more than one part, write a plan instead —
   `skill://clip-editing/editing-intelligence.md` — because a plan is where
   the story is decided, and a sequence placed by hand has none: the check
   before a render says so. By hand, for a trim of one recording:
   - A clip's `start` and `end` are the `source_range`. To leave a little air
     around it, extend by up to `safe_in` before and `safe_out` after. Those
     are measured from the silence around the clip, so anything within them
     cannot clip a word — and anything past them can. Do not add a margin of
     your own and do not work the numbers out from the silences yourself.
   - Merge kept parts that are less than 0.3 s apart into one clip, so the
     result is not choppy.
   - Build the sequence with `insert_clip` in source order.
   - Never cut on round seconds picked by eye (5 to 25, 10 to 30). A cut goes
     where a sentence ends or a pause falls, and the only way to know where
     that is, is to read the clips.
   - Leave out clips whose `kind` is `unusable`, and say so if the user asks
     for a stretch that is mostly those.
8. For subjective selections such as highlights, list the chosen parts with
   times and quoted text, and confirm with the user before rendering.
9. Speech recognition invents sentences over shots with nobody in them — a
   channel sign-off, a subtitle credit — and returns them at high confidence.
   A stretch measured as silent is never `speech`, whatever text came back,
   and such a line is not offered as a caption either, so both `query_clips`
   and `generate_subtitles` already keep them out. What the rule cannot catch
   is a mishearing of something that was said: check a quote that matters
   against the surrounding clips, and ask the user about a name.

Detection, transcription and rendering all run on the user's machine and no
media file is uploaded anywhere. Frames and transcripts do reach you as tool
results, though, so if you are not running on that machine they have left
it. Say so plainly if the user asks, and before putting private footage
through `frames_for_clips`.

## Requests about content

| Request | Operations |
|---|---|
| 「這支影片在講什麼」 / what is this video about | `analyze_asset`, `build_semantic_timeline`, then summarize what `query_clips` returns with times, and `view_frames` for the visuals |
| 「剪掉講錯／重講的地方」 / remove flubbed takes | In a plan the compiler already keeps only the last complete go at a repeated line. By hand: from the transcript, keep only the last complete version of each repeated sentence |
| 「去掉停頓／氣口」 / remove pauses | In a plan it is done for you. By hand: `query_clips` for `kind: speech`, then place them as consecutive clips, each widened by its own `safe_in` / `safe_out` |
| 「只留有講到 X 的段落」 / keep only parts about X | `query_clips` with `text`, keep those sections plus enough context to make sense |
| 「剪掉黑畫面／畫面卡住的地方」 / remove black or frozen parts | Keep the time outside `black_frames` / `frozen_frames`; a plan does this at every cut on its own |
| 「找出精華剪成 60 秒」 / a 60 s highlight reel | Choose segments by transcript and frames until about 60 s; confirm the list before rendering |
| 「第 3 分鐘那個畫面是什麼」 / what is on screen at 3:00 | `view_frames` with one `asset_ids` entry, `start: 175`, `end: 185`, `count: 4` |

## When the user brings raw footage and no plan

Someone who has never edited will hand you a folder and say 「幫我剪一下」.
Do not ask 「你想剪什麼主題？」: not knowing is why they asked. Look at the
footage yourself, then give them a choice between concrete directions.

Work through it in three passes, so the slow one runs only on what survives:

1. **Survey, seconds per file.** `import_folder` on the folder, or
   `import_asset` per file, then `view_frames` over the whole list, about 6
   frames each, six files to a call — not one call per file. This alone tells
   you how much footage there is, which files have sound, what each one
   shows, and which are unusable because they are dark, shaky, or a stray
   recording.
2. **Triage, one decoding pass per file.** On the files that look usable,
   call `analyze_asset` with `transcribe: false`. That finds scene changes,
   silences, and black or frozen picture, measures each shot's exposure, blur,
   motion and shake along with the sound's levels, and looks for faces — all
   without running speech recognition, which is the slow part. Use it to find
   where anything happens and to drop files that are mostly nothing.
3. **Propose, then transcribe.** Offer two or three directions and let the
   user pick. Only once they have picked, run `analyze_asset` with
   transcription, on the files that direction needs. Build the semantic
   timeline over them afterwards, and select from it.

Never transcribe a whole folder up front. Twenty ten-minute files take hours,
and most of it gets thrown away.

A proposal says what is in the footage, how long each option runs, and what it
leaves out. Each option is a story, not a list of places: say what it opens
on, where it turns, and what it comes to. 「每個地點各留一點」 is not an
option; it is the fragmented cut nobody wants to watch. Read `skill://clip-editing/pacing-and-structure.md` before naming
those lengths. Label the options so the user can answer with one letter:

「這 8 支我都看過了，共 42 分鐘。裡面有三條線：海邊 4 分鐘、晚餐聊天有講到之後的行程 6 分鐘，
另外 2 支畫面晃得很厲害，建議不要用。
A：「計畫被打亂的一天」，約 90 秒。開在海邊的好天氣，轉在突然下雨躲進餐廳，收在晚餐聊到的下一趟行程。
B：只留晚餐那段對話，約 2 分鐘，比較完整：從「下次去哪」開始，吵了一輪，最後決定的地方收尾。
你想要哪一種？」

When the user answers 「你決定」, take the first option, say in one line which
one you took, and show them a storyboard before you render.
