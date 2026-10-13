# Handling feedback

Read this when the user has watched a version and says what is wrong with
it — 「太長」「開頭無聊」「音樂太大聲」「剛剛那樣比較好」 — or asks what changed
since the last one. It is referenced from SKILL.md.

## Change the plan, not the timeline

When the user asks for a change, change the plan and compile it again rather
than nudging clips on the timeline. The plan is where the reasons live; the
timeline is only what fell out of them. Anything the user adjusted by hand is
pinned and survives the recompile.

Change it with `amend_plan`, one amendment per thing that actually changed,
and pass the user's own words as `note` — they are kept with the version, and
they are what makes the history readable later. Do not send the whole plan
through `save_plan` again to move one edge: every other selection's reason is
retyped to do it, and those reasons are the part that cannot be rebuilt.

- One piece: `set_trim` to retrim it, `set_playback` to speed it up or turn
  it down, `set_rationale` to rewrite why it is
  there or move it to another beat, `add_selection` to put something in,
  `drop_selection` to take something out, which files it under `rejected`
  with the reason. `add_broll` and `drop_broll` do the same for covering
  picture.
- Between two parts: `set_beat_join` puts a transition or a J-cut on the
  beat that comes in.
- Words in the plan, only when the user asked: `set_beat_title` opens a
  part on a title card, `set_selection_texts` puts a name, a place or a
  headline over a piece. Cards and words added by hand stay through a
  recompile.
- The whole video: `set_pacing` makes every cut tighter or looser at once —
  a lower `pause_seconds` takes out more dead air between sentences — a
  pause inside a sentence is how somebody talks, and stays — a lower `breath_seconds`
  leaves less around every cut, and `speed` plays the whole sequence faster
  with voices at their own pitch (1.1 is rarely noticed as speed). `set_music_level` turns the music by a
  `scale`, everywhere or from one `beat_id`. `set_target` changes the length
  or platform the cut is held to. `drop_beat` takes out a whole part and files
  everything in it under `rejected`.

Then `validate_plan`, `compile_plan` into the project, and look again.

## Show what changed

After a round, call `preview_plan_diff` with the plan twice — `before_version`
the one before the round, `after_version` the one after. It is one sheet of the
new cut with added shots framed green, retrimmed yellow, moved blue, the ones
taken out in red at the end, and a line on how the length changed, part by
part. Describe it in a sentence or two rather than reading the list out:
「開頭拿掉兩句、結尾那段移到前面，整支短了 12 秒。」 `diff_plan` gives the same
comparison as text, down to the reasons.

## Going back

Every change to a video is kept as a version: a compile, an edit, captions,
whoever made it. 「剛剛那樣比較好」 or 「回到昨天那版」 is `project_history` to
find the one they mean — each says when, who, and what changed in words — and
`restore_version` to bring it back. That is a new version on top, so going
back can be undone the same way and nothing is lost. The plan the video was
compiled from goes back with it; when another video shares that plan, this
one gets its own copy, and `plan_copied` says so — tell the user.

「做一個 Reels 版」 or 「試試沒有配樂的」 is a branch: `branch_project` from the
version to start at, with a name in the user's words. It is a project of its
own beside the first, which is left alone.

A version can carry marks: `starred` and `published`, which the user sets in
the editor's version panel or asks for (「把這版標起來」「這版發到 IG 了」 is
`mark_version`), and `exported`, which a version rendered out wears by itself.
「回到我標星號那版」 or 「回到發出去那版」 is the version with that mark.
「這版跟第 3 版差在哪」 is `compare_versions`: what changed, part by part, in
sentences you can read back as they are. 「開頭用第 3 版的，其他用現在的」 is
`merge_version_parts` with those parts picked from the older one — a new
version on top, with the plan merged by beat when both came from one. The
user can do the same in the editor's comparison view.

The user can also leave comments in the editor while watching: a moment or
a stretch of the cut with what they want there. Anything you do to that
video says `open_comments` when some are waiting — read them with
`get_comments`, which says where each is on the cut now (it follows the
footage it was written on; `gone` means that footage was cut out), do what
each asks like any other feedback, then `resolve_comment` with what you did
in a sentence the user reads. Ask with `resolve_comment` and `resolved:
false` when one can be read two ways. Comments are the user's to delete.

Pass the user's words for every change as `note` — on `apply_edits`,
`compile_plan`, `generate_subtitles`, `amend_plan` — since that is what the
version says in the history the user reads. A plan's own versions are still
there too: `list_plan_versions`, `revert_plan`, and `get_plan` with a
`version`.

## What the complaint means

Complaints are about how it feels, and more than one change fits most of
them. Take the one in this table first, say in one line what you changed, and
let the user steer from there.

| The user says | Change |
|---|---|
| 「太長了」 / too long | `set_target` if they name a length; then drop the weakest selections with `drop_selection`, whole parts with `drop_beat`, before retrimming the rest. Take dead air out before content (`set_pacing`) |
| 「太短」「講不完整」 / too short, cut off | `set_trim` back to `full` on the pieces that stop mid-thought; `validate_plan` notes a cut that still ends mid-sentence |
| 「整體節奏太慢」「很拖」 / it drags | `set_pacing` with a lower `pause_seconds` (0.45) and `breath_seconds` (0.05); still slow, add `speed: 1.1`; if it still drags, it is the content — drop selections |
| 「太趕」「喘不過氣」「剪太碎」 / too rushed, too choppy | `set_pacing` with more air: `breath_seconds` 0.2 and a higher `pause_seconds` (0.9) |
| 「這段不要」 / lose this bit | `drop_selection` with their reason |
| 「這整段不要」 / lose this whole part | `drop_beat` with their reason |
| 「開頭無聊」 / the opening is boring | No amendment does this for you: it is choosing footage. Move the liveliest moment first (`set_rationale` with `beat_id`, or `add_selection` with `before_clip_id`), and drop the slow opening |
| 「結尾拖」 / the ending trails off | `set_trim` with `head` on the last piece, ending on the last line that lands |
| 「那段講到 X 的怎麼沒放」 / why is the part about X missing | Read `rejected` for the reason; `add_selection` it back if they still want it |
| 「音樂太大聲」「蓋過講話」 / the music is too loud | `set_music_level` with `scale: 0.6`, then `preview_sound` to check |
| 「音樂太小聲」 / the music is too quiet | `set_music_level` with `scale: 1.5` |
| 「聽不太清楚」 / hard to hear | `preview_sound` first. Noise under the voice: `set_clip_audio` with `cleanup`. Music over it: `set_music_level`. A voice left apart from the others: its file was not transcribed |
| 「畫面太悶」「一直同一個畫面」 / the picture is dull | `propose_broll`, then `add_broll` |
| 「字太小」「字被擋住」 / captions too small or covered | `set_caption_style` with the platform's `preset`, or a larger `size_fraction` |
| 「剛剛那樣比較好」「回到上一版」 / the last one was better | `project_history`, then `restore_version` to that version |
| 「改了哪裡？」 / what changed | `preview_plan_diff` with the plan and the two versions |
| 「這段我自己調的不要動」 / keep what I did here | Already kept: a hand edit pinned it |

When a complaint could mean two quite different things — 「怪怪的」, 「不太對」 —
ask which part, with two or three concrete guesses to choose from.
