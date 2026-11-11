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
   of — each with its `role` in the story, below — and a `selection` per piece of footage naming the beat it belongs to
   and the `rationale` for it being there. Put what you considered and passed
   over into `rejected` with the reason. That field earns its keep the moment
   the user asks 「那段講到 X 的怎麼沒放」.
2. Each selection's `trim` says how much of its clip to use: `full`, `keep`
   with the sentences inside a section to keep, `head` or `tail` with a number
   of seconds, `range` with `from_seconds` and `to_seconds` for a stretch out
   of the middle of a long shot, or `tighten` to drop the pauses out of a
   section. There is no free-text trim. If none of these says what you mean, select `utterance`
   clips instead and name them one by one — the plan drops to the level it
   needs rather than leaving the decision to something downstream.
   How a piece plays is part of the plan too: `speed` on a selection (4 or 8
   for a timelapse of a walk, 0.5 for slow motion) and `volume` (0 mutes a
   shout or the wind under music). How one part hands over to the next is on
   the beat: `transition_in` (`dissolve`, `dip`, `wipe`) and `sound_lead` for
   a J-cut, where the next place is heard before it is seen. Put them here
   rather than on the timeline afterwards: music, markers and covering
   picture are laid out against the plan's lengths, and a transition the
   footage has too little picture for is shortened with a note. A beat whose
   first shot starts at the top of its file has no picture before it at all:
   the plan is refused, with how much later to start that shot, the dip that
   needs half as much, or taking the transition off — the choice is yours,
   since starting later cuts that much of the shot.
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
6. When there are two ways to cut the same footage, `copy_plan` the first,
   change the copy, and use `diff_plan` to tell the user what actually
   differs. `copy_plan` with a `timeline_id` also carries a plan onto a
   timeline built over a different set of files, where the same moments have
   other clip IDs — or over the same files transcribed again, where every
   sentence edge moved a little. Read `matched_nearby` back to the user: a
   sentence that was split or joined may now say more or less. `get_plan` reads one
   back, with the other plans listed for comparison.

Compiling again is safe. A compiled clip remembers which plan and which
footage it came from, and any clip you adjust by hand — trimmed, moved,
recoloured, split — is pinned by that edit alone. A pinned clip comes through
the next compile with its adjustment intact, only in whatever place the new
plan gives it. So do not redo a hand adjustment after recompiling: it is still
there, and the result says how many clips were `kept` that way. Music,
markers and covering picture are laid against the pinned clips as they now
play, so after changing a clip's length or speed by hand, compile again to
move them with it. A clip lengthened by hand over the footage of the one after
it takes that footage in: the next compile does not put the other back. If a
recompile is refused because a pinned clip's footage is no longer in the plan,
that is a real choice to put to the user: either the clip goes back into the
plan, or `set_clip_pinned` with `pinned: false` hands it back and the next
compile rebuilds it. Do not unpin their work without asking.

The plan owns the sequence track, the covering picture and the music bed and
rebuilds all three. Insets and any other track you added are untouched by a
compile.

## The structure

A video is held together by something, not a run of moments. Before choosing
footage, decide what the video is about in one sentence — something that
changes, not a place or a day — and which structure carries it. **The
structure is the user's to choose.** Unless they have already said, offer the
two or three that suit the footage, each in one line with what it would open
on and end on, and let them pick by letter, in the same message as the other
open questions. Do not pick one silently and do not offer them all. Put the
choice in the plan's `structure`; it decides which `role` each beat may have.

| `structure` | 中文 | Roles, in the order they play | Suits |
|---|---|---|---|
| `story` (default) | 起承轉合 | `hook` `setup` `turn` `payoff` | A vlog, a day, anything where something happened |
| `flashback` | 倒敘 | `hook` `setup` `turn` `payoff`; the hook is a moment from later | When the ending or the key moment is the strongest thing shot |
| `problem_solution` | 問題→解法 | `hook` `problem` `agitate` `solution` `result` | Explainers, a product, fixing a pain point |
| `listicle` | 清單式 | `hook` `item`… `recap` | 「N 個…」: tips, picks, places — one point a beat |
| `tutorial` | 教學步驟 | `hook` `prep` `step`… `result` | Cooking, making, software: show the result, then each step |
| `before_after` | 前後對比 | `hook` `before` `process` `after` | Makeovers, tidying, repairs |
| `review` | 開箱評測 | `hook` `first_look` `detail`… `verdict` | Unboxing, trying out, comparing |
| `montage` | 蒙太奇／氛圍 | `hook` `build` `peak` `outro` | Travel, events, scenery cut to music, with no plot |
| `ad` | 廣告／產品介紹 | `hook` `problem` `product` `proof` `cta` | Ads, sponsored segments, product introductions |

What each role is for:

- `hook` — the first seconds: the strongest moment, or the question the rest
  answers. Optional in a very short cut except in 倒敘 and 廣告, and always first when
  there is one. In 倒敘 it is lifted from the turn or the payoff, and the
  setup then goes back to where it began; in 教學步驟 and 前後對比 it is a
  glimpse of the finished thing.
- 起承轉合: `setup` is what the viewer needs to follow — who, where, what was
  meant to happen; `turn` is where something changes: it goes wrong,
  surprises, gets decided. If the footage seems to have none, look for the
  failure, the wait, the change of plan; ask the user what happened that day
  rather than inventing it. `payoff` is what it came to, and the video ends
  there, not on whatever was shot last.
- 問題→解法: `problem` is what is wrong, `agitate` why it matters or what it
  costs, `solution` what fixes it, `result` what it looks like fixed.
- 清單式: one `item` beat per point, at least two, and an optional `recap`.
  Put the strongest point first or last, not in the middle.
- 教學步驟: `prep` is what you need, one `step` beat per step, `result` the
  finished thing. Keep the steps in the order they are done.
- 前後對比: `before`, then `process` (optional, often sped up), then `after`.
  Match the framing of before and after where the footage allows.
- 開箱評測: `first_look` is the first impression, a `detail` beat per thing
  looked at closely, `verdict` whether it is worth it.
- 蒙太奇: `build` beats rise towards a `peak`, there can be more than one of
  each, and `outro` lets it settle. The music carries it, so choose the song
  first and consider `cut_on_beat`.
- 廣告: each role is a slot from `propose_sections(structure="ad")`, and its
  beats play exactly the clips `set_sections(slots=...)` stored for it, in
  that order; the reading-footage file says how. `hook` and `product` are
  required, and it ends on the product, the proof or the cta.

`validate_plan` refuses a plan with more than one beat or more than one file
that has a beat with no role, a role its structure has no place for, none of
a role the structure needs (a turn, a step, a peak), or a last beat the
structure cannot end on. 清單式, 教學步驟, 前後對比, 開箱評測 and 廣告 also have to play
their roles in the order above. A single part cut from one recording is a
trim and needs none. When the user changes structure after a plan is
written, rewrite the beats' roles with it — the old ones are refused, not
translated.

Choose each selection for what it does in its beat, and say so in its
`rationale`: a shot that is only pretty, or only there because it was shot,
goes in `rejected`. Within a beat, keep what somebody says whole — a
sentence cut in half is worse than a sentence left out — and let a beat run
as long as its point needs rather than giving every file an equal share.

Keep a thread running through the whole video, not only through each beat:
every beat should move the one-sentence idea on, and a piece that is only
there because it happened that day is the piece viewers call 「不連貫」.
`validate_plan` refuses the commonest version of that — a piece dropped
between two that belong together: shot hours away from both of them, or
about another topic than both. Move it next to what it belongs with, or drop
it; only when the jump is the point — a flash-forward, a callback — keep it
and say so in its `jump_reason`. A hook and the first piece of a beat may
jump freely, since that is how a part begins.

A reason can be written for anything; what the viewer knows cannot. Before
choosing footage for a plan of more than one part, list in `facts` what the
viewer has to come to know for it to make sense — 「水管接頭裂了」 before
「修好了」 — one short `id` and a sentence each. Then give each piece that
shows or says one an `establishes` entry with its `evidence`: the seconds of
the file where it happens, found by looking at or listening to that stretch,
not from the clip's summary. A piece that only makes sense once something is
known `requires` it. `validate_plan` refuses a fact relied on before it is
established, a fact nothing establishes, and evidence the trim cuts off. The
hook may rely on what comes later; anywhere else, a result shown before its
cause on purpose lists the cause in `ahead`, and it has to come. A shot there
only for the mood establishes nothing, and that is fine.

## Covering picture (B-roll)

A second pass, after the rough cut is right. `propose_broll` says where the
picture holds too long — one shot held on, or several in a row from the same
file. Then find something that belongs over those words with `query_clips`,
and add a shot with `add_broll`: the `clip_id` it plays from, the
`over_clip_id` it starts over, and how many `seconds` it runs. The sound
underneath keeps running, which is the whole point. The picture plays from the
start of `clip_id`; when the moment to show is partway into a long take — the
mechanism working, not the hands reaching for it — find it with `view_frames`
and set `from_seconds` to where it begins.

The server enforces the rest and refuses with a reason: a shot runs 1 to 6
seconds, no more than 40% of a beat may be covered, two shots may not overlap,
none may run over a title card, and a selection marked `hold_picture` may not
be covered at all. Set
`hold_picture` on the shots where what the speaker is doing has to be seen —
pointing at something, holding something up. Nothing measures that, so if you
do not say it, it is not known.

### One camera, nothing to cover it with

A long stretch of one shot — someone talking to a single camera — comes back
from `propose_broll`, and `query_clips` may find nothing that belongs over it.
Do not cover it with another moment of the same take; that is the same picture
again. Change the picture inside the shot instead, roughly in this order:

1. Shorten it: dead air first (`set_pacing`), then whole sentences
   (`drop_selection`, or a tighter `trim`).
2. Punch in. A compiled stretch is already several clips wherever a pause was
   cut out; give every other one `set_clip_look` with `fit: {"mode": "fill",
   "zoom": 1.2}` and a `center_x`/`center_y` on the face, so each jump cut
   reads as a cut to a closer shot. `split_clip` a long clip with no pause to
   make one. Keep it near 1.2–1.3 on footage shot at the output size, or it
   goes soft, and check with `view_frames`. That is a hand edit, so it is pinned
   and survives a recompile, but `propose_broll` does not see it and still
   lists the stretch.
3. Words over it, where the user asked for them: a headline over the line that
   matters (`set_selection_texts`).

A slow push is the other way to change the picture inside one shot: `motion:
"push"` closes in 8% over the clip, keeping the face where it is. In a
compiled cut set it with `amend_plan` `set_motion`, so a recompile keeps it;
every clip compiled from the piece pushes over its own length, which with the
punch-in above gives the usual alternation of a push and a held closer shot.
It is refused on a clip under 1.5 s, where it reads as a jolt. `check_render`
lists under `moves` the clips it softens, footage shot at the output size
enlarged at the end of the push: tell the user rather than dropping it.

## Music in a plan

Every video gets music unless the user declines, so a `save_plan` with no
`music` needs their no. Say it in the read-back (「我會配兩段音樂：前段輕快，
成功之後換溫暖的」) and drop it if they decline. When they said to finish on
your own, add it without asking and report the songs, and any credit they need,
with the finished video. Only a library with nothing usable stops this: ask for
a file the way the main skill says. `find_music` cannot find a song that is
still `analyzing`, so analyze the songs and look again before saying none fits.

Change the song where the feeling turns, such as a failure becoming a success
or the work giving way to the ending. Never change it by the clock, and leave
one song playing where nothing turns. These ceilings are Provisional, not
targets: one song up to 60 seconds, two or three up to 3 minutes, and about one
for every 1–2 minutes past that. Each part gets its own song unless coming back
to one is the point.

Music runs from the first second to the last, ducked under speech. A cue with
no `asset_id` leaves a part bare, so use it only when the user asked, or when
you named that part and why in the read-back.

For each part, `find_music` with what that part wants and its length as
`min_seconds`, and put the closest song in its cue so the preview has music.
You cannot hear the songs and the match is a guess, so then `offer_songs` with
that part's `beat_id`, once for every part, before telling the user: each part
keeps its own songs, and they hear them in whatever order they like. Offering a
part again replaces only that part's songs. Tell them once, from the last
call's `waiting`: the songs are in the editor, or they can answer here by part
and number. A pick in the editor comes back as a comment and is already in the
cue; one given in the conversation you write yourself, with `amend_plan`
`set_music_cue` on that part's `beat_id` (left out for the part that starts the
video) and the song's `asset_id` and `start`. When they said to finish on your
own, there is nobody to pick, so skip the offers.

A plan's `music` is a list of `cues`. Each one names the `beat_id` it comes in
on — the first may leave it out to start with the video — and plays until the
next cue takes over: 「開場輕快、訪談段不要音樂、結尾換一首」 is three cues, the
middle one with no `asset_id`. `start` is where in the song to come in; a song
shorter than its part loops back to there. The fades are per cue, so two songs
meet end to end, one fading out as the other fades up. For the next song to
overlap the last, set `crossfade_seconds` on the music: it comes in that much
early and rises while the other fades, and is still exactly where its cue
says at the cut. A song coming in from its very top has nothing earlier to
come in with, so there the one going out runs on past the cut instead. Use
2 seconds (Provisional: no length has been compared by ear) unless the user
says otherwise.

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

## Sound effects in a plan

The moments come from the compiled cut, so compile first, then
`propose_sound_effects`. Its `candidate_hash` changes when a shot, a card,
words or B-roll comes or goes, and `compile_plan` then refuses the old
picks: after any such amendment, propose again and pick again before you
compile. Pick from the effects' `sound_kinds`, which only you or the user
gave; a short file imported without them shows none, so pass them to
`import_asset` from the file's name. A `riser` builds into its moment: give
it a `lead` of its own length, or it starts at the cut and swells after it.
Write the picks with `amend_plan` `set_sound_effects`.

## Cleaning up the sound

**A narration over footage is handled for you.** When a voice recorded apart
from the picture — a voice-over, a clip-on microphone's own file — sits on an
audio track, the render brings it up to a standard level and drops the
footage's own sound (the street, the café) and any music while it speaks. A
track counts as a voice when its recording was transcribed and sentences
cover at least half of it, so analyze the narration file like any other; a
track set to duck under speech is music and never counts. `set_track_audio` with
`voice` says so outright either way. Do not turn the footage down by hand for
this, and do not mute it: the place under a narration is part of the video.
When the footage also carries someone talking to camera, both are kept, and
the narration wins where they overlap. `preview_sound` puts the narration on
the voice side and the place on the other, so you can see it sits under.

Two further things, both off unless asked for.

`set_clip_audio` with `cleanup` repairs one clip's voice: `rumble` takes out
the low roar of traffic or air conditioning, `hiss` the steady background a
phone leaves, `sibilance` the harsh S sounds a close microphone picks up. Each
one works by throwing part of the recording away, which is why none of them is
a default and why the settings are gentle — a voice scrubbed until it sounds
underwater is a worse result than the hiss was. Ask which problem the user
actually hears rather than turning all three on.

Levels are set for you where the footage is transcribed. The talking in every
clip is brought to one level, measured over its sentences alone, so somebody
far from the camera and somebody shouting into it come out alike; a clip
nobody talks in — a street, a café, a timelapse — is kept under that talking
rather than over it; the music sits against the talking too. So do not even
out voices by hand. A whisper or a shout that is meant to be one is the
exception: `keep_level` on that selection leaves it where it was recorded
against the rest. A clip's `volume` still applies on top of all this, so 0.5
still means half as loud as its neighbours.

Softening the sound across a transition is not automatic, and deliberately so:
how long and which side are judgements. Use `audio_fade_out` on the outgoing
clip and `audio_fade_in` on the incoming one when the user asks for it.

## Requests about the cut

| Request | Operations |
|---|---|
| 「這段畫面太悶，蓋點別的」 / cover this with other footage | `propose_broll`, then `amend_plan` with `add_broll` |
| 「這段畫面不能蓋掉」 / this shot has to be seen | `hold_picture: true` on that selection |
| 「這個人聲音小很多」「兩個人音量差很多」 / match two people's levels | Already matched where the footage is transcribed; `preview_sound` to check. If one stays apart, it was not transcribed — analyze it with transcription |
| 「這句悄悄話要小聲」「這裡大叫要保留」 / keep this one as loud or quiet as it was | `amend_plan` with `set_playback` and `keep_level: true` on that selection |
| 「背景很吵，人聲不清楚」 with a narration on its own track / the voice is buried | `preview_sound` first. The footage should already sit under the narration; if the narration is not on the voice side, it was not recognised as one — `set_track_audio` with `voice: true` |
| 「有雜音」「背景很吵」「嘶嘶聲」 / clean up a voice | `set_clip_audio` with `cleanup`: `rumble` for the low roar, `hiss` for the background, `sibilance` for harsh S sounds |
| 「每一段換不同的音樂」「訪談那段不要音樂」 / change the music between parts | One music cue per part, each with the `beat_id` it comes in on; a cue with no `asset_id` is silence |
| 「剪接跟著音樂節拍」 / cut to the beat | `cut_on_beat` on that music cue, after `analyze_asset` on the song. Ask first — it suits a montage, not an interview |
| 「這段我自己調的不要動」 / keep my version of this one | Already kept: editing it by hand pinned it. `set_clip_pinned` `pinned: true` says so for a clip nobody has touched |
| 「這段照計畫重做就好」 / rebuild this one from the plan | `set_clip_pinned` with `pinned: false`, then compile again |
| 「這段快轉」「這段靜音」 / speed up or mute one piece | `amend_plan` with `set_playback`: `speed` and `volume` for that selection |
| 「章節之間溶接」「先聽到下一段」 / a transition or J-cut into a part | `amend_plan` with `set_beat_join` on the beat it comes into |
| 「中間那段就好」 / only the middle of this shot | `set_trim` with `range` |

Transcripts can misrecognize names and jargon. When a quote matters, check the
surrounding segments, and ask the user if the meaning is unclear.
