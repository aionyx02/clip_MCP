# Requests and the operations they turn into

Read this when a request does not plainly name what to do. It is referenced from
SKILL.md, which says how the work goes; this is the lookup from what people say to
the calls it takes.

Times like `1:30` mean 90 seconds. Unless the user refers to the edited result,
times refer to the source file. Requests about content, the plan, delivery and
feedback are mapped in their own files.

| Request | Operations |
|---|---|
| 「把 a.mp4、b.mp4 接起來」 / join these videos | `insert_clip` for each file in order, `source_range` `{start: 0, end: duration}` |
| 「a.mp4 只要 10 到 20 秒」 / use 0:10–0:20 of a.mp4 | `insert_clip` with `source_range` `{start: 10, end: 20}` |
| 「剪掉 X 開頭 3 秒」 / cut the first 3 s of X | `trim_clip` X, `new_source_range` `{start: start + 3, end: end}` |
| 「剪掉 X 最後 3 秒」 / cut the last 3 s of X | `trim_clip` X, `new_source_range` `{start: start, end: end - 3}` |
| 「刪掉第二段」 / delete the second clip | `delete_clip` (later clips move up); with `ripple: false` to keep the gap |
| 「在 A 和 B 中間插入 c.mp4」 / insert between A and B | `insert_clip` with `before_clip_id: B` |
| 「把 C 移到最前面」 / move C to the front | `reorder_clip` C with `before_clip_id` = the first clip; without it, to the end |
| 「把這段從中間切開」 / split this clip | `split_clip` with `at` (timeline) or `at_source` (source), plus a `new_clip_id` for the second half |
| 「改成倒敍，結尾放最前面」 / open with the ending | `split_clip` at the boundary, then `reorder_clip` the tail with `before_clip_id` = the first clip |
| 「把 A 換成 d.mp4」 / replace A with d.mp4 | `insert_clip` new clip with `before_clip_id: A`, then `delete_clip` A |
| 「A 和 B 中間停 2 秒黑畫面」 / 2 s of black before B | `move_clip` B and every later video clip, each with `new_timeline_in` set to its current `timeline_in` plus 2 |
| 「音樂從第 10 秒才進來」 / start the music at 0:10 | `move_clip` the first music clip with `new_timeline_in: 10`, then `fit_track` again |
| 「用歌的 1:05 開始」 / start from the chorus at 1:05 | Music clip `source_range.start: 65` |
| 「音樂小聲一點／大聲一點」 / music quieter or louder | `set_clip_audio` on every music clip, `volume` × 0.6 or × 1.5 (in a plan: `set_music_level`) |
| 「音樂淡出」「音樂比影片長／短」 / fade the music, fit it | `fit_track` with `fade_out: 2` |
| 「拿掉背景音樂」 / remove the music | `delete_clip` every clip on the audio track |
| 「人聲出現時音樂小聲一點」 / duck the music under the talking | `set_track_audio` on the music track with `duck_under_speech: true` |
| 「找一首輕快的歌」「配首溫暖的鋼琴」 / find a song | `find_music` with it in English and the part's length as `min_seconds`; read the top few back with how each sounds |
| 「這首歌是講話，不是音樂」 / move a file between footage and music | `set_asset_library`, and say so |
| 「從 Pixabay 抓這首」 / take a song from the web | `add_music_from_url` without `confirm_plan`, tell the user its licence and credit, then with it once they agree |
| 「說明欄要標什麼出處」 / credits for the description | `music_credits` |
| 「回到我標星號那版」 / back to the starred version | `project_history`, the version with `starred` in its `marks`, then `restore_version` |
| 「把素材分資料夾」 / sort the library | `organize_library`, then say what was made and filed |
| 「把影片原音關掉」 / mute the original sound | `set_clip_audio` `volume: 0` on every video clip |
| 「聲音先進來」「上一句講完再切」 / J cut, L cut | `set_clip_audio` with `audio_lead` on the incoming clip, or `audio_lag` on the outgoing one |
| 「每段音量差很多」 / the volume jumps between clips | Nothing where the footage is transcribed: every clip's talking is brought to one level and the places nobody talks in are kept under it. Otherwise `render_project` normalizes the finished mix |
| 「這裡用溶接」「不要硬切」 / cross dissolve | `set_clip_look` on the incoming clip with `transition_in: {kind: "dissolve", seconds: 1}` |
| 「用擦劃轉場」「從左邊掃過去」 / wipe | `set_clip_look` with `transition_in: {kind: "wipe", seconds: 0.6, direction: "left"}` |
| 「這裡淡到黑再進來」「過白場」 / dip through a colour | `set_clip_look` with `transition_in: {kind: "dip", seconds: 1, through: "black"}`; `white` or a hex colour also work |
| 「轉場拿掉，改回硬切」 / back to a straight cut | `set_clip_look` with `clear_transition: true` |
| 「開頭淡入、結尾淡出」 / fade in and out | `set_clip_look` `video_fade_in` on the first clip, `video_fade_out` on the last |
| 「這段快轉」「放慢一點」 / speed it up or slow it down | In a compiled cut, `amend_plan` with `set_playback`, so what is laid over it moves too. Otherwise `set_clip_speed` with `speed`; 2.0 is twice as fast. Voices keep their pitch unless `preserve_pitch: false` |
| 「這段亮一點／色彩濃一點」 / brighter or more colourful | `set_clip_look` with `color` `brightness` or `saturation`; ones you leave out keep their value |
| 「改成黑白」 / black and white | `set_clip_look` with `color` `{"saturation": 0}` |
| 「色溫暖一點／冷一點」 / warmer or cooler | `set_clip_look` with `color` `temperature`, below 6500 for warmer |
| 「調色拿掉」 / undo the grade | `set_clip_look` with `clear_color: true` |
| 「這段整個畫面都要看到」「直式裡橫的畫面被切掉了」 / show a shot whole | `set_clip_look` with `fit: {"mode": "whole"}` |
| 「這段往左一點」「拉近一點」 / move or zoom the crop | `set_clip_look` with `fit: {"mode": "fill", "center_x": 0.3}` or `"zoom": 1.5` |
| 「放這張照片」 / put a photo in | `insert_clip` with `source_range` `{start: 0, end: 4}`; longer when asked |
| 「開頭加個標題」「每段前面加標題卡」 / add a title card | `add_title_card` with `title` (and `subtitle`), `before_clip_id` the shot it leads into; 2.5 s unless `seconds` says |
| 「標題卡改成黑底／用這張照片當底」「標題留久一點」 / change a card | `set_title_card` with `background: "colour"` and `colour`, `photo_asset_id`, `seconds`, `title` or `subtitle` (`""` removes it) |
| 「這裡打上他的名字」「標一下這是哪裡」 / a name or a place | `add_text` on the clip with `style: "name"` (`second`: their role) or `"place"`, `timeline_start`/`timeline_end` where it should show |
| 「這句話用大字打出來」 / a headline | `add_text` with `style: "headline"` over that stretch |
| 「字移到右上」「名字晚一點出來」「字拿掉」 / change words | `set_text` with `style: "free"`, `x`, `y`; or `timeline_start`; `remove_text` |
| 「照片不要動」「照片慢慢拉遠／往左移」 / how a photo moves | `set_clip_look` with `motion`: `none`, `pull`, `pan_left`, `pan_right`, or `push` (the default) |
| 「右上角放一個小視窗」 / an inset in the top right | `add_track` a second video track, then `add_clip` with `timeline_in` and a `layout` box |
| 「小視窗拿掉」 / drop the inset | `delete_clip` it, or `set_clip_look` with `clear_layout: true` to cover the frame |
| 「中間插一段別的畫面蓋掉原本的」 / cut away over the same sound | `add_clip` on the upper video track with no `layout` and `volume: 0` (in a plan: `add_broll`) |
| 「這段我自己調的不要動」 / keep my version | Already kept: a hand edit pinned it. `set_clip_pinned` `pinned: true` for an untouched clip |
| 「標一下開場到哪裡」 / mark where a part begins | `set_markers`; `compile_plan` already writes one per beat |
| 「這支叫 EP1 台北」 / name this project | `rename_project` with the new `name` |
| 「做成直式／方形」 / vertical or square | `render_project` with `frame`; see the platform file |
| 「加字幕」 / add captions | `generate_subtitles`, proofread the lines it returns, fix them with `edit_subtitle`, then render with `burn_subtitles: true` |
| 「這段沒講話也上個字」 / text over a shot | `add_subtitle` with the shot's `asset_id`, `source_start`, `source_end` in that file and `text`; it shows even on a muted clip |
| 「字幕有個字打錯了」 / a caption has the wrong word | `get_subtitles` around that moment for its `cue_id`, then `edit_subtitle` with the new `text` |
| 「後來又加了一段，那段沒有字幕」 / new footage has no captions | `generate_subtitles` again; the lines already corrected come back as corrected |
| 「這句字幕多停一下」 / hold this caption longer | `edit_subtitle` with a new `source_end` |
| 「這句不要了」 / drop this caption | `edit_subtitle` with `delete: true` |
| 「跟上一版差在哪」 / what changed between versions | `project_history` for the commits, then `compare_versions` |
| 「開頭用舊的那版」 / take a part from another version | `compare_versions` for the part names, then `merge_version_parts` with those parts as `a` |
| 「照我的留言改」 / do what my comments say | `get_comments`, change each moment it names, then `resolve_comment` with what was done |
| 「現在剪成什麼樣子」 / show me the cut so far | `preview_project`, then describe the order and the cut points |
| 「先聽聽看」 / let me hear it | `preview_sound`, and give the user the mix file |

"The second clip" means the second video clip ordered by `timeline_in`. For a
time in the edited result, such as 「成片第 12 秒」, find the clip where
`timeline_in <= 12 < timeline_in + (end - start)`. The source time is
`start + (12 - timeline_in)`.
