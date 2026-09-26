# Deciding what goes in the cut

Read this when you are choosing which footage to use, in what order, and what
goes over it: writing a plan, covering picture with B-roll, putting music
under parts of the video, or fixing how it sounds. It is referenced from
SKILL.md. How long each part should run is in
`skill://clip-editing/pacing-and-structure.md`; what to change after the user
has watched it is in `skill://clip-editing/feedback-handling.md`.

## Write a plan, not a timeline

For anything longer than a handful of clips, write a plan before you touch the
timeline. A plan says what the video is for, what its parts are, which footage
fills them and why — and the server works out every second from it: the air
around each cut, the dead air taken out, repeated takes dropped, cuts moved
off words and bad frames. That is what lets the user change their mind later
without the whole thing being redone from memory.

1. `save_plan` with a `goal`, a `target` length, the `beats` the video is made
   of, and a `selection` per piece of footage naming the beat it belongs to
   and the `rationale` for it being there. Put what you considered and passed
   over into `rejected` with the reason. That field earns its keep the moment
   the user asks 「那段講到 X 的怎麼沒放」.
2. Each selection's `trim` says how much of its clip to use: `full`, `keep`
   with the sentences inside a section to keep, `head` or `tail` with a number
   of seconds, or `tighten` to drop the pauses out of a section. There is no
   free-text trim. If none of these says what you mean, select `utterance`
   clips instead and name them one by one — the plan drops to the level it
   needs rather than leaving the decision to something downstream.
3. `validate_plan` before compiling. It costs nothing and catches a clip that
   is not there, a beat nothing belongs to, a trim longer than its clip, and
   footage marked `unusable` slipping in. `problems` stop the plan; `notes`
   are worth reading — a cut a third longer than the length asked for is a
   note, not an error.
4. Read the plan back to the user in plain language — how many parts, how long
   each runs, why each is there — before compiling. They can say 「不用看，
   直接剪」 and skip it, but that is theirs to skip, not yours to assume.
5. `compile_plan` into an empty project builds the whole cut. Compiling the
   same plan again always gives the same cut, so an edit can be reproduced.
   A project that already has clips is refused: make a new one and keep both.
6. When there are two ways to cut the same footage, save two plans and use
   `diff_plan` to tell the user what actually differs. `get_plan` reads one
   back, with the other plans listed for comparison.

Compiling again is safe. A compiled clip remembers which plan and which
footage it came from, and any clip you adjust by hand — trimmed, moved,
recoloured, split — is pinned by that edit alone. A pinned clip comes through
the next compile with its adjustment intact, only in whatever place the new
plan gives it. So do not redo a hand adjustment after recompiling: it is still
there, and the result says how many clips were `kept` that way. If a
recompile is refused because a pinned clip's footage is no longer in the plan,
that is a real choice to put to the user: either the clip goes back into the
plan, or `set_clip_pinned` with `pinned: false` hands it back and the next
compile rebuilds it. Do not unpin their work without asking.

The plan owns the sequence track, the covering picture and the music bed and
rebuilds all three. Insets and any other track you added are untouched by a
compile.

## Covering picture (B-roll)

A second pass, after the rough cut is right. `propose_broll` says where the
picture holds too long — one shot held on, or several in a row from the same
file. Then find something that belongs over those words with `query_clips`,
and add a shot with `add_broll`: the `clip_id` it plays from, the
`over_clip_id` it starts over, and how many `seconds` it runs. The sound
underneath keeps running, which is the whole point.

The server enforces the rest and refuses with a reason: a shot runs 1 to 6
seconds, no more than 40% of a beat may be covered, two shots may not overlap,
and a selection marked `hold_picture` may not be covered at all. Set
`hold_picture` on the shots where what the speaker is doing has to be seen —
pointing at something, holding something up. Nothing measures that, so if you
do not say it, it is not known.

## Music in a plan

A plan's `music` is a list of `cues`. Each one names the `beat_id` it comes in
on — the first may leave it out to start with the video — and plays until the
next cue takes over: 「開場輕快、訪談段不要音樂、結尾換一首」 is three cues, the
middle one with no `asset_id`. `start` is where in the song to come in; a song
shorter than its part loops back to there. The fades are per cue, so two songs
meet with one fading out as the other fades up.

`cut_on_beat` on a cue moves every picture cut under it onto the song's beat.
It is a style, not a fix: right for a montage or a fast short, wrong for an
interview, so ask rather than turning it on. It needs the song analyzed —
`analyze_asset` on the music file finds its beat — and it only ever moves a
cut through silence, a fraction of a second at most, so it never cuts into a
word. `validate_plan` says how many cuts moved and how many could not reach a
beat; a cut that could not stays where it was. Footage that was never
transcribed is never moved, because nothing says where its words are. A song
with no steady pulse, such as an ambient pad, has no beat to cut on, and the
plan is refused with that reason.

Whether the music actually gets out of the way of the talking cannot be seen
on a storyboard, and you cannot hear it: `preview_sound` shows it.

## Cleaning up the sound

Two separate things, both off unless asked for.

`set_clip_audio` with `cleanup` repairs one clip's voice: `rumble` takes out
the low roar of traffic or air conditioning, `hiss` the steady background a
phone leaves, `sibilance` the harsh S sounds a close microphone picks up. Each
one works by throwing part of the recording away, which is why none of them is
a default and why the settings are gentle — a voice scrubbed until it sounds
underwater is a worse result than the hiss was. Ask which problem the user
actually hears rather than turning all three on.

`level_voices` on the plan is for a conversation where one person was much
louder than the other. `validate_plan` says how far apart the voices are when
it is worth saying, so read that first: under a few decibels there is nothing
to fix. It turns the louder people down to match the quietest, so nothing
clips, and the render's own loudness normalization brings the whole thing back
up afterwards. A window where two people talk over each other is left alone —
it belongs to neither of them.

Softening the sound across a transition is not automatic, and deliberately so:
how long and which side are judgements. Use `audio_fade_out` on the outgoing
clip and `audio_fade_in` on the incoming one when the user asks for it.

## Requests about the cut

| Request | Operations |
|---|---|
| 「這段畫面太悶，蓋點別的」 / cover this with other footage | `propose_broll`, then `amend_plan` with `add_broll` |
| 「這段畫面不能蓋掉」 / this shot has to be seen | `hold_picture: true` on that selection |
| 「這個人聲音小很多」「兩個人音量差很多」 / match two people's levels | `save_plan` or `amend_plan` with `level_voices: true`; `validate_plan` says how far apart they are |
| 「有雜音」「背景很吵」「嘶嘶聲」 / clean up a voice | `set_clip_audio` with `cleanup`: `rumble` for the low roar, `hiss` for the background, `sibilance` for harsh S sounds |
| 「每一段換不同的音樂」「訪談那段不要音樂」 / change the music between parts | One music cue per part, each with the `beat_id` it comes in on; a cue with no `asset_id` is silence |
| 「剪接跟著音樂節拍」 / cut to the beat | `cut_on_beat` on that music cue, after `analyze_asset` on the song. Ask first — it suits a montage, not an interview |
| 「這段我自己調的不要動」 / keep my version of this one | Already kept: editing it by hand pinned it. `set_clip_pinned` `pinned: true` says so for a clip nobody has touched |
| 「這段照計畫重做就好」 / rebuild this one from the plan | `set_clip_pinned` with `pinned: false`, then compile again |

Transcripts can misrecognize names and jargon. When a quote matters, check the
surrounding segments, and ask the user if the meaning is unclear.
