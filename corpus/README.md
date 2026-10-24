# 語料庫

L2 評測的題目。一份 `.json` 是一題：**素材 + 指令 + 標註**，格式見
`src/app/benchmark/case.py`，計分見 `metrics.score`。

目標是 20–30 支、維度鋪開：單人 vlog、對話、吵雜戶外、講話底下有音樂、
多機混拍、中英夾雜。其中一部分標成 `holdout`，只用來驗證，不參與任何常數
或規則的調整。

`corpus/*.json` 不進 git。標註指向的素材只存在某一台機器上，內容又引用了
當事人自己的需求與對自己作品的說法，不是這個 repo 該公開的東西。進 git 的
只有這份 README；案例本身由每個人用自己的素材種出來。

## 素材根目錄

標註裡的路徑一律是**相對於語料根目錄**的，不是絕對路徑，也不是 asset ID——
asset ID 每個工作區各自產生，拿它當錨點的語料換一台機器就什麼都量不到。

語料根目錄是你放素材的資料夾，底下是 `raw/<日期>/` 與 `music/`。
計分時由呼叫端把相對路徑接上根目錄、匯入成 asset，再把
`{相對路徑: asset_id}` 交給 `score`。

## ⚠️ 成片跟素材放在同一個資料夾

`raw/<日期>/` 底下每個 `<日期>.mp4`（`0828.mp4`、`0902.mp4`…）**是成片，不是素材**。
它們都是 30fps、都匯入過、都不在任何語意時間軸裡。

所以 `footage` 千萬不要用「列資料夾」的方式組——那等於把答案交給被測的那一端。
`seed` 也一樣：檔案是一個一個點名的，不接受資料夾。**做完的成片本來就不該放在
素材根目錄**；`render_project` 會寫到使用者影片資料夾下的 `clip-mcp/`，找得到，
不必靠每次記得避開。

## 一題有哪些欄位

- `id`、`instruction`、`footage`：題名、當初的指令、素材的相對路徑。
- `instruction_by`：`user`（本人寫的）、`summarised`（本人寫的、別人看過成片後縮短）、
  `derived`（看過成片才反推的）。愈靠後面愈不像原本的問題。
- `annotated_by`：`user` 或 `draft`。`draft` 的分數不算證據，`score` 會把它寫進
  `Scorecard.unmeasured`。
- `footage_files`：每個素材的 `sha256`、`bytes`、`duration`，以及 `source`
  （怎麼拿到這個檔案：哪台相機、哪顆硬碟、哪裡下載）。舊案例沒有這個欄位，
  代表素材是「相信它」而不是「驗過它」。
- `kinds`：這是哪種素材，自由文字。
- `split`：`tune` 或 `holdout`。`holdout` 只在最後驗證時看，不能拿來調門檻。
- `target_seconds`、`must_keep`、`must_drop`、`cut_windows`：長度與標註。
  `cut_windows` 空的不算違規，計分器會當成「沒人畫過範圍」，而不是「每個剪點都錯」。
- `transcripts`：每支素材的人工校對逐字稿（字與時間）。`checked_by` 是 `user`
  才會拿去算 WER／CER；`draft` 是辨識器自己的輸出，只拿來當校對的起點。
- `caption_lines`：字幕應該在哪裡斷行，寫給之後的計分用。
- `rubric`：L3 判官用的，這一層不計分。

## 雜湊與驗證

案例把素材的大小與 sha256 釘住。`file_digest(path)` 一次讀 1 MiB 算出這兩個數字；
`verify_footage(case, root)` 回傳每個對不上的檔案一行原因（不見了、大小變了、
雜湊變了），全部相符時回傳空清單。

`score(case, pieces, analyses, asset_ids, root=...)` 會在有 `footage_files` 時
呼叫它。對不上就整題寫進 `unmeasured`——素材不對，量到的東西就不是這題的答案。
沒給 `root` 也一樣算「沒驗過」，不會當成通過。

WER 用 `tokens` 切：中文一個字一個 token，英文一個詞一個 token，所以中文的 WER
其實是字元錯誤率；CER 則一律以字元計。兩者都寫在 `transcript_score` 回傳的
`{file: {"wer", "cer", "words"}}` 裡。

## 種一份草稿

```bash
python -m app.benchmark.seed \
  --root D:/video_work --case-id 0828-trip \
  --instruction "剪成一分鐘左右，開頭用在大安森林公園那段" \
  raw/20260828/A.MP4 raw/20260828/B.MP4
```

- 算好 `footage_files`（雜湊、大小；`duration` 讀得到就讀）。
- 工作區裡如果已經有同一路徑的分析，就把逐字稿複製成 `checked_by: "draft"`。
- 寫成 `annotated_by: "draft"`、`split: "tune"`、`instruction_by: "user"`。
- 預設寫到目前資料夾的 `corpus/`，用 `--corpus` 換位置；同名檔案已存在就拒絕，
  要覆蓋得加 `--force`。`--split holdout`、`--instruction-by summarised` 可以改。

## 種完之後

1. **把 `must_keep` 砍到只剩真的非留不可的。** 草稿是從既有成片反推的嗎？那就會把
   用過的每一段都標成必留，量出來的是「跟舊版有多像」，正好是這個語料庫不要量的東西。
   一段一段看 `why`，把「選了」跟「非選不可」分開。
2. **補 `must_drop` 與 `cut_windows`。** 空的不是錯，但有的題目才分得出好剪點與壞剪點。
3. **校對逐字稿。** `checked_by` 改成 `user` 才會算 WER／CER；沒校對的稿子當 draft。
4. **改完把 `annotated_by` 改成 `user`。** 在那之前這題一分都不算。
5. **挑幾題標成 `holdout`。** 調過的東西不要再拿同一題回頭驗，否則那個數字只是在背答案。
