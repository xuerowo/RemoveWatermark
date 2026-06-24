# 去水印工具

這是一個本機使用的去水印工具，可以用「水印模板」偵測固定樣式的半透明水印，接著用 LaMa 補圖模型把遮罩區域補起來。

LaMa 是去水印時用的補圖模型。第一次執行時，相關套件可能會自動下載模型檔，所以第一次會比較久。

## 專案資料夾

- `src/remove_watermark/`：主要程式碼。
- `remove_watermark/`：讓 `python -m remove_watermark` 可以在本機執行的入口。
- `input/`：預設輸入圖片資料夾。
- `templates/`：水印模板資料夾。
- `output/`：預設輸出圖片資料夾，通常不放進版本管理。
- `tmp/`：本機暫存檔，通常不放進版本管理。
- `scripts/`：安裝與輔助腳本。
- `tests/`：自動測試。
- `tests/fixtures/input/`：測試用固定範例圖片。

## 安裝環境

第一次使用請先建立 `.venv`，它是 Python 的本機執行環境。

最簡單的安裝方式是雙擊：

```cmd
setup-venv.bat
```

雙擊後會出現選單，可以選：

1. 基本環境，不安裝 SAM3
2. 基本環境加 SAM3/AI 工具，不下載模型
3. 基本環境加 SAM3/AI 工具，並下載 SAM 3.1 模型

SAM 3.1 模型約 3.5GB，所以選第 3 項時第一次會比較久。

也可以手動執行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-venv.ps1
```

手動執行上面這行只會安裝基本環境。如果也要安裝 SAM3/AI 工具並下載模型：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-venv.ps1 -InstallAiTools -DownloadAiModels
```

安裝腳本會自動偵測 NVIDIA GPU。RTX 50 系列會使用 CUDA 12.8 版 PyTorch；沒有 NVIDIA GPU 的電腦會使用 CPU 版。你也可以指定：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-venv.ps1 -TorchBuild Cuda128
powershell -ExecutionPolicy Bypass -File scripts\setup-venv.ps1 -TorchBuild Cpu
```

`.bat` 包裝檔也可以轉交參數；只要有帶參數，就會略過選單並直接轉交給 PowerShell 腳本：

```cmd
setup-venv.bat -TorchBuild Cpu -SkipSmokeTest
setup-venv.bat -InstallAiTools -DownloadAiModels
```

如果只想基本安裝，雙擊 `setup-venv.bat` 後選第 1 項，或直接執行上面的 `scripts\setup-venv.ps1`。

如果只想安裝執行必要套件，不安裝測試工具和程式檢查工具：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-venv.ps1 -SkipDevTools
```

## 執行測試

日常開發預設跑快速測試，會排除較慢的真圖片回歸、慢速測試和完整資料集掃描：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

日常提交前也建議跑靜態檢查：

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

如果要檢查偵測品質相關的真圖片回歸測試：

```powershell
.\.venv\Scripts\python.exe -m pytest -m image_regression -n auto
```

慢速偵測評估可以依目的分開跑。`false_positive` 是誤判防線，`false_negative` 是漏判防線，`cross_dataset` 是跨資料集檢查，`performance` 是較大範圍或較花時間的檢查：

```powershell
.\.venv\Scripts\python.exe -m pytest -m "slow and false_positive" -n auto
.\.venv\Scripts\python.exe -m pytest -m "slow and false_negative" -n auto
.\.venv\Scripts\python.exe -m pytest -m "slow and cross_dataset" -n auto
.\.venv\Scripts\python.exe -m pytest -m "slow and performance" -n auto
```

發版前或大幅調整偵測演算法時，請至少跑完整非 exhaustive 測試：

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not exhaustive" -n auto
```

如果這次改到偵測門檻、遮罩邏輯或測試素材，發版前也要另外跑圖片回歸測試：

```powershell
.\.venv\Scripts\python.exe -m pytest -m image_regression -n auto
```

完整跨資料集掃描只在大幅調整演算法或準備重要版本時執行：

```powershell
.\.venv\Scripts\python.exe -m pytest -m exhaustive -n auto
```

## 演算法評估報表

調整偵測門檻或遮罩策略前，建議先建立評估 manifest。Manifest 是 JSON 格式的案例清單，包含圖片、模板和預期框：

```json
{
  "cases": [
    {
      "image": "tests/fixtures/local_input/ai_clean_01.png",
      "templates": ["tests/fixtures/templates/watermark_black.png"],
      "expectedBoxes": [
        { "bbox": [756, 281, 360, 96] }
      ]
    }
  ]
}
```

Manifest 內的相對路徑會以 manifest 檔案所在資料夾為基準。`expectedBoxes` 可以只填 `bbox`；如果同一個案例使用多個模板，則每個預期框都必須加上 `template`，才能分模板統計命中與漏判。

產生 JSON 與 CSV 報表：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_detection.py --manifest eval-manifest.json --output-json output\eval-report.json --output-csv output\eval-report.csv
```

JSON 報表會包含整體摘要、每張圖片、每個模板的候選數、去重後候選數、偵測數、耗時、TP/FP/FN 和每個偵測框的最佳 IoU。CSV 報表會攤平成每模板/每偵測一列，方便用試算表排序與篩選。TP 是命中，FP 是誤判，FN 是漏判，IoU 是偵測框與預期框的重疊率。

如果要分析候選為何被接受或拒絕，可以加上候選診斷：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_detection.py --manifest eval-manifest.json --output-json output\eval-report.json --include-candidates
```

候選診斷會寫在 JSON 的 `candidateDiagnostics`，包含候選來源、尺度、`score`、`gray_score`、`edge_score`、`color_score`、接受/拒絕狀態，以及擋下候選的 guard 原因。

## 模板偵測 benchmark

要比較不同模板數、尺度設定與圖片尺寸的偵測耗時，可以使用 benchmark 工具。專案內有一份小型範例：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_detection.py --manifest tests\fixtures\benchmark_detection_manifest.json --output-json output\benchmark-detection\summary.json --output-csv output\benchmark-detection\summary.csv
```

Manifest 可設定 `variants` 與 `cases`。`variants` 控制 `minScale`、`maxScale`、`scaleStep` 與 `repeats`；`cases` 控制圖片、單模板/多模板，以及可選的 `maxSide` 縮圖尺寸。JSON 報表會輸出每個組合的 `medianElapsedMs`、`meanElapsedMs`、候選數、去重候選數、偵測數、圖片尺寸、模板數與尺度數，並依 variant 與模板數彙整；摘要中的 `meanElapsedMs` 會依 `repeats` 加權。CSV 報表方便用試算表比較前後版本。

`output/` 已被 `.gitignore` 排除。benchmark 產物請留在本機比較，不要提交大型輸出。

## 遮罩品質評估

調整遮罩 body 或膨脹策略前，可以用 mask evaluation 產生視覺 artifact。Manifest 格式和偵測評估相同：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_masks.py --manifest mask-eval-manifest.json --output-dir output\mask-eval
```

每個案例會依 manifest 順序輸出到 `001_<檔名>` 這類資料夾，避免同檔名圖片互相覆蓋。內容包含偵測框、原圖/結果對照、LaMa mask 疊圖、mask body 疊圖、body source、mask body、LaMa mask、cleanup change（變更熱區）與 patch compare（局部比較），並寫入 `summary.json` 和各案例的 `state.json`。

如果要評估實驗性的 SAM3 refine 是否值得保留，可以在有 CUDA/SAM3 的環境加上比較參數：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_masks.py --manifest mask-eval-manifest.json --output-dir output\mask-eval-sam3 --compare-sam3-refine
```

專案內也有一份低對比黑色水印的範例 manifest：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_masks.py --manifest tests\fixtures\sam3_refine_eval_manifest.json --output-dir output\sam3-refine-eval --compare-sam3-refine
```

這會在每個案例的 `state.json` 增加 `sam3RefineComparison`，並在 `summary.json` 增加 `sam3Refine` 摘要。重點欄位包含 baseline/refined 遮罩像素數、交集/聯集、IoU（遮罩重疊率）、refined-only、missed-baseline、baseline/refine 耗時與失敗原因。摘要中的 `failureRate` 只計算實際嘗試 SAM3 refine 的案例；沒有偵測框的 `skipped` 案例會另外計數，不會稀釋失敗率或耗時中位數。只有在這份報表顯示遮罩品質或人工修正量明顯改善，且失敗率可接受時，才應把 SAM3 refine 納入正式功能。

目前本機報表在 3 個代表案例中 `attemptedCount=3`、`failureCount=3`、`failureRate=1.0`，原因是缺少 Meta `sam3` 套件；因此 SAM3 refine 不升格為正式功能，維持預設關閉的實驗選項。這個結論限於本機缺少 `sam3` 的環境，不代表已驗證真實 SAM3 遮罩品質。

## Google Colab 使用

如果想在 Google Colab 上使用，請打開：

```text
RemoveWatermark_Colab.ipynb
```

Colab 是 Google 的雲端 Python 筆記本。建議在 Colab 上方選單 `Runtime -> Change runtime type` 選 GPU，然後照筆記本順序執行。更詳細的步驟請看：

```text
COLAB.md
```

## 啟動 Web UI

最簡單的方式是雙擊：

```cmd
start-ui.bat
```

也可以手動執行：

```powershell
.\.venv\Scripts\python.exe -m remove_watermark.web
```

啟動後會自動開啟瀏覽器到：

```text
http://127.0.0.1:8765
```

如果不想自動開瀏覽器，可以加上：

```powershell
.\.venv\Scripts\python.exe -m remove_watermark.web --no-open-browser
```

Web UI 支援：

- 新增單張圖片或匯入資料夾。
- 匯入水印模板或模板資料夾。
- 從目前遮罩建立新模板。
- 單張偵測水印。
- 批量偵測水印。
- 單張去水印。
- 批量去水印。
- 可用單張或批量的「一鍵去水印」直接連續完成偵測與去水印。
- 可下載目前圖片的去水印結果，或把所有結果打包成 ZIP 壓縮包。
- 可選擇去水印後是否保留偵測框，預設不保留。
- 遮罩筆刷、橡皮擦、還原筆刷。
- 復原與重做。
- 原圖、結果、並排、前後對比滑桿檢視。
- 處理結果會先暫存在 Web UI 工作區；需要正式取回時，請用「下載目前圖片」或「下載全部圖片」。

預設情況下，Web UI 只會寫入工作狀態和可下載的處理結果暫存檔。如果想保留 UI 的除錯資料，可以用 debug 模式：

```powershell
.\.venv\Scripts\python.exe -m remove_watermark.web --save-debug
```

除錯資料會存到：

```text
output/debug-ui/<圖片名稱>/
```

裡面會包含手動畫的遮罩、遮罩疊圖、偵測框疊圖、輸入/輸出比較圖、變更區域圖和狀態 JSON。

## 批量處理速度設定

Web UI 的批量偵測預設會依照 CPU 數量平行處理。批量去水印預設一次處理一張，這樣比較不容易把 GPU 或記憶體吃滿。

可以用下面參數調整：

```powershell
.\.venv\Scripts\python.exe -m remove_watermark.web --batch-detect-jobs 4 --batch-process-jobs 1
```

- `--batch-detect-jobs 0`：自動使用 CPU 數量。
- `--batch-process-jobs`：同時跑幾張 LaMa 去水印。只有在 GPU 或 CPU 記憶體夠大時才建議調高。

## GPU 和 CPU

LaMa 是目前唯一的去水印後端，使用 `simple-lama-inpainting`。

預設是：

```text
--lama-device auto
```

意思是：

- 如果 PyTorch 偵測到 CUDA 可用，就優先用 NVIDIA GPU。
- 如果沒有 GPU，就用 CPU。
- 如果是自動模式選到 GPU，但 LaMa 載入或執行失敗，會自動退回 CPU。
- 如果你明確指定 `cuda`，就會維持 GPU 模式，不會自動退回 CPU。

強制使用 CPU：

```powershell
.\.venv\Scripts\python.exe -m remove_watermark --lama-device cpu
```

## AI 偵測水印

除了原本的模板偵測，也可以用 `SAM3` 模式自動找水印。這個模式會直接用 SAM 3.1 的文字提示找水印位置和遮罩，最後仍然交給 LaMa 補圖。不需要訓練或微調，但第一次使用會下載模型，時間會比較久。

如果要重新安裝環境並包含 AI 工具：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-venv.ps1 -InstallAiTools
```

如果也要順便下載 SAM 3.1 模型：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-venv.ps1 -InstallAiTools -DownloadAiModels
```

目前 AI 偵測只保留這個模型：

- `facebook/sam3.1`（ModelScope）

只下載模型、不處理圖片：

```powershell
.\.venv\Scripts\python.exe -m remove_watermark --download-ai-models
```

只下載並快取 ModelScope 的 SAM 3.1 模型：

```powershell
.\.venv\Scripts\python.exe -m remove_watermark --download-sam3-model
```

使用 SAM 3.1 文字提示直接找水印位置和遮罩：

```powershell
.\.venv\Scripts\python.exe -m remove_watermark --detector sam3 --ai-prompt "watermark. text watermark. logo." --input input --output output
```

Web UI 右側的「偵測模式」切到 `SAM3` 後，可以調整文字提示、信心門檻、最大面積和最多區域。

如果想調整 AI 找什麼，可以改提示文字：

```powershell
.\.venv\Scripts\python.exe -m remove_watermark --detector sam3 --ai-prompt "watermark. logo. text watermark. stamp."
```

AI 偵測可能會漏掉很淡的水印，也可能把圖片裡真正的文字或商標誤判成水印。建議先用 Web UI 看遮罩，必要時再用遮罩筆刷修一下。

如果 AI 抓到過大的區域，可以調低提示範圍，或調整 `--ai-box-threshold` 和 `--ai-max-box-area-ratio`。

模板模式的進階參數裡也有「SAM3 修正遮罩」實驗開關。它會先用模板找水印框，再請 SAM3 依框細修遮罩；預設關閉，且 SAM3 失敗或回傳空遮罩時會自動回退模板遮罩。

SAM 3.1 模型檔約 3.5GB，載入時也會比較吃記憶體和時間。這個整合目前需要 NVIDIA CUDA GPU。它需要 Meta 官方 `sam3` 程式碼和 `modelscope` 套件；如果環境還沒安裝 AI 工具或還沒下載模型，請重新執行 `setup-venv.bat` 並選第 3 項，或手動執行 `scripts\setup-venv.ps1 -InstallAiTools -DownloadAiModels`。

另外，SAM 3.1 的 `.pt` 權重是 PyTorch 模型檔格式，請只載入可信來源的模型。不要隨便改成陌生網站下載的 `.pt` 檔。

## 命令列模式

使用預設資料夾執行：

```powershell
.\.venv\Scripts\python.exe -m remove_watermark
```

預設資料夾是：

- `input/`：輸入圖片。
- `templates/`：水印模板。
- `output/`：輸出圖片。

如果想指定路徑：

```powershell
.\.venv\Scripts\python.exe -m remove_watermark --input input --template templates --output output
```

命令列模式也可以輸出除錯資料：

```powershell
.\.venv\Scripts\python.exe -m remove_watermark --save-debug
```

除錯資料會存到：

```text
output/debug/
```

每次偵測會記錄送進 LaMa 的擴張遮罩 `lama_mask`。`text_detail_mask` 也會保留給人工檢查，但不會改變送進 LaMa 的遮罩。

## 遮罩設定

送進 LaMa 前，程式會從水印模板的輪廓建立遮罩。輪廓會稍微閉合成一整塊，但會限制擴張面積，避免變成單純的大矩形。如果輪廓建立失敗，會退回使用閉合後的主體遮罩。

Web UI 模板模式的「SAM3 修正遮罩」是實驗功能，只在手動打開時使用；它需要 SAM3 / CUDA 環境，失敗時仍會保留模板遮罩。

如果想調整主體連接的積極程度：

```powershell
.\.venv\Scripts\python.exe -m remove_watermark --mask-body-gap-ratio 0.06 --save-debug
```

固定膨脹像素會再受 `mask body` 最短邊比例限制，避免小水印被 10px 膨脹吃進太多背景。預設比例是 `0.10`；如果要更保守或關閉這個相對上限：

```powershell
.\.venv\Scripts\python.exe -m remove_watermark --mask-dilate-max-body-ratio 0.08 --save-debug
.\.venv\Scripts\python.exe -m remove_watermark --mask-dilate-max-body-ratio 0 --save-debug
```

如果想關掉整個 `mask body` 主體合併：

```powershell
.\.venv\Scripts\python.exe -m remove_watermark --no-mask-body
```
