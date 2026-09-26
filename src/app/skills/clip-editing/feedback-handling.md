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

- One piece: `set_trim` to retrim it, `set_rationale` to rewrite why it is
  there or move it to another beat, `add_selection` to put something in,
  `drop_selection` to take something out, which files it under `rejected`
  with the reason. `add_broll` and `drop_broll` do the same for covering
  picture.
- The whole video: `set_pacing` makes every cut tighter or looser at once —
  a lower `pause_seconds` takes out more dead air, a lower `breath_seconds`
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

Every save and every amendment is a version, and all of them are kept.
「剛剛那樣比較好」 is `list_plan_versions` to find the one they mean — each
version has its note and what it changed — and `revert_plan` to bring it back.
That saves the old version as a new one on top, so going back can itself be
undone and nothing is lost either way. `get_plan` with a `version` reads one
without going back to it. Compile again afterwards.

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
| 「聽不太清楚」 / hard to hear | Two people at different levels: `level_voices`. Noise under the voice: `set_clip_audio` with `cleanup`. Music over it: `set_music_level` |
| 「畫面太悶」「一直同一個畫面」 / the picture is dull | `propose_broll`, then `add_broll` |
| 「字太小」「字被擋住」 / captions too small or covered | `set_caption_style` with the platform's `preset`, or a larger `size_fraction` |
| 「剛剛那樣比較好」「回到上一版」 / the last one was better | `list_plan_versions`, then `revert_plan` to that version, then `compile_plan` |
| 「改了哪裡？」 / what changed | `preview_plan_diff` with the plan and the two versions |
| 「這段我自己調的不要動」 / keep what I did here | Already kept: a hand edit pinned it |

When a complaint could mean two quite different things — 「怪怪的」, 「不太對」 —
ask which part, with two or three concrete guesses to choose from.
