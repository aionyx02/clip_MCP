# Pacing and structure

Read this when you are deciding how long the edit should be, what to open
on, or how fast to cut. It is referenced from SKILL.md.

These are defaults for when the user has no opinion of their own. Drop any of
them the moment they say otherwise.

**The story card comes before the plan.** On a production, `set_story` writes the
paper edit: the video in one sentence, its structure, how it opens, where it
turns, how it ends, what is left out, and its rough length. Write every part so
the user could say it is wrong — name the moments in this footage, 「開場是印歪的
第一個杯架，轉在杯子終於卡進去」, not 「輕鬆有趣」. In `interactive` show it in the
conversation and `confirm_story` with its `story_hash` and the user's own words
before `save_plan`: a plan saved first is cut to a story nobody agreed to. In
`auto` write it and go on; `get_production`'s `pending` keeps it first until the
user says yes. Never `confirm_story` in your own words.

**The parts have a structure.** Which one, and what each role is for, is in
`skill://clip-editing/editing-intelligence.md`. Pacing serves it. In 起承轉合
and 倒敘 the setup can breathe, the turn is where the cutting tightens, and
the payoff gets a beat of silence after it rather than the next thing. In 清單式
every item runs about as long as the others, so none feels skipped. In 教學步驟 a
step runs as long as it takes to see it done, and waiting is sped up rather
than cut out. In 前後對比 the after holds longer than the before. In 蒙太奇 the
shots get shorter towards the peak and longer again in the outro.

**Open on the strongest moment.** The first two or three seconds decide
whether the rest gets watched, so do not open on a title card, a slow pan, or
someone settling into frame. Find the liveliest moment and put it first with
`reorder_clip`; if it sits in the middle of a clip, `split_clip` there first
and lift the piece you want.

**Cut on the content, not the clock.** In short vertical video most shots run
about 1.5 to 3 seconds and rarely past 5, and something should change on
screen every few seconds. In a talking-head piece, cut when a sentence ends
rather than mid-thought, and leave a beat after the person lands a point. In
longer horizontal video shots can run far longer; do not chop it up out of
habit.

**Take out the dead air before taking out content.** Most raw footage
tightens by roughly a third just by removing pauses, restarts, and the
seconds before and after someone speaks. A plan does that at every cut on its
own, and `set_pacing` makes it stricter across the whole video. Only when it
is still too long should you start dropping things the user might have wanted.

**End on purpose.** Do not let it trail off on someone reaching for the
camera. Trim the last clip to the last meaningful frame, and put the music
fade-out there.

**Respect the frame.** Output is cropped to fill, so in a vertical project
most of the width of a landscape source is gone. The crop follows the face
the analysis found, but only the largest one, and anything else off to the
side is lost; on a phone the platform's own captions and buttons cover part
of the frame as well (`skill://clip-editing/platform-conventions.md`). Check
`preview_project` instead of assuming, and say so when something important
falls outside.

Length is a decision, not a default. Ask for it unless the user said, and
name a number when you propose one: 「短一點」 gives you nothing to cut to.
