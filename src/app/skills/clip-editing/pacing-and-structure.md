# Pacing and structure

Read this when you are deciding how long the edit should be, what to open
on, or how fast to cut. It is referenced from SKILL.md.

These are defaults for when the user has no opinion of their own. Drop any of
them the moment they say otherwise.

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
