# 去水印工具

這是一個本機使用的去水印工具，可以用「水印模板」偵測固定樣式的半透明水印，接著用 LaMa 補圖模型把遮罩區域補起來。

LaMa 是去水印時用的補圖模型。第一次執行時，相關套件可能會自動下載模型檔，所以第一次會比較久。

## 專案資料夾

- `src/remove_watermark/`：主要程式碼。
- `remove_watermark/`：讓 `python -m remove_watermark` 可以在本機執行的入口。
- `scripts/setup-venv.ps1`：PowerShell 安裝腳本。
- `setup-venv.bat`：Windows 雙擊安裝入口。
- `start-ui.bat`：Windows 雙擊啟動 Web UI。
- `RemoveWatermark_Colab.ipynb`：Google Colab 筆記本。
- `COLAB.md`：Google Colab 使用說明。

下面這些是本機工作資料夾，會被 `.gitignore` 排除，不會放進公開倉庫：

- `input/`：預設輸入圖片資料夾。
- `templates/`：水印模板資料夾。
- `output/`：預設輸出圖片資料夾。
- `tmp/`：本機暫存檔。

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

如果只想安裝執行必要套件，不安裝開發輔助工具：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-venv.ps1 -SkipDevTools
```

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
