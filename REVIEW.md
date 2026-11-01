# REVIEW.md

Instructions for the automated reviewer. Keep it short. When the reviewer (or you) keeps making the same comment, move it here.

## What failure looks like here

- A render or a preview quietly comes out wrong: the wrong stretch of footage, sound off the picture by even a few milliseconds, the wrong loudness, captions on the wrong words. A test that passes without measuring the output is no evidence.
- The server deletes, moves or overwrites a file of the user's library or footage, or writes outside the workspace.
- The server decides for the model or the user, or quietly fixes what it should refuse and explain (the design principles in `CLAUDE.md`).

## Always review deeply

- `.github/workflows/**`, `installer/**`: CI, the release and the Windows installer.
- `src/app/storage/**`: history, trash and cache clean-up; these delete files.
- `src/app/engine/builder.py`, `loudness.py`, `playback.py`, `sync.py`: what the render and the preview hear and see, and when.
- `src/app/engine/renderer.py`, `resources.py`, `models.py`: job workers, memory admission, models kept between jobs.
- `src/app/engine/fetch.py`, `src/app/ui/**`: downloads from URLs and the local web server; paths and names from outside.

## Skip

- `uv.lock`
- `src/app/ui/static/*.png`, `*.ico`

## Conventions to enforce

- The design principles in `CLAUDE.md`. In particular:
  - a threshold or constant without evidence is marked `Provisional` with what it measures;
  - no compatibility layers or migrations; data whose meaning changed is made again;
  - a check that fails refuses and says why, it does not quietly fix.
- A test for timing or sound measures the real output (FFmpeg on synthesized media), not the command line.
- What leaves the machine (thumbnails, transcripts in tool results) is said plainly in the README and the skill.

## Reporting

- Cite file:line as evidence, never a guess from a name.
- At most 3 Nits per round.
- The summary starts with a count and says plainly when there is nothing to fix.
- Language: English
