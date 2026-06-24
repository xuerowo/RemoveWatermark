# Google Colab 使用說明

這個專案可以在 Google Colab 跑。Colab 是 Google 提供的雲端 Python 筆記本，適合沒有本機 NVIDIA GPU，或想先用雲端 GPU 測試的人。筆記本可以啟動 Web UI，並支援 LaMa 補圖、模板偵測和 SAM3 偵測。

## 建議用法

1. 打開 `RemoveWatermark_Colab.ipynb`。
2. 在 Colab 上方選單選 `執行階段 -> 變更執行階段類型`，把硬體加速器改成 `GPU`。
3. 依序執行筆記本裡的儲存格。
4. 如果專案還沒放到 GitHub，可以把整個專案壓成 zip 上傳到筆記本第一格。
5. 第二格會安裝套件，預設也會下載 SAM 3.1 模型。模型約 3.5GB，第一次會比較久。第二格也會預設啟用 Google Drive 快取，用來減少之後重複下載。
6. 網頁介面啟動後，用筆記本輸出的 Colab 視窗開啟工具。
7. 圖片和水印模板可以直接在 Web UI 裡上傳，不一定要先放進 Colab 資料夾。

## 專案來源選項

筆記本第一格有三種方式。畫面上會顯示中文選項，程式內部仍會轉成穩定的英文代碼：

- `上傳 zip`：上傳專案 zip，最適合目前這份本機專案。
- `GitHub`：填入 GitHub 專案網址後自動下載。
- `Google Drive 資料夾`：從 Google Drive 裡已存在的專案資料夾複製。

## 注意事項

- 第一次執行 LaMa 會下載模型檔，所以會比較久。
- SAM3 需要 CUDA GPU。Colab 沒有分配到 GPU 時，請在 Web UI 切到「模板」模式，或在批量模式選「模板」。
- SAM 3.1 模型約 3.5GB。筆記本第二格的「下載_SAM3_模型」預設為開啟；如果只想用模板或 LaMa，可以先關掉，這樣會略過 SAM3 套件安裝和模型下載。之後要用 SAM3 時，再打開並重新執行第二格。
- 筆記本第二格的「使用_Google_Drive_快取」預設為開啟，會把 `pip` 下載快取、ModelScope、Hugging Face 和 Torch 的快取放到 `/content/drive/MyDrive/RemoveWatermark/.cache`。這不是永久安裝；runtime 重開後仍需重新執行安裝格，但會盡量重用已下載的檔案。
- 不建議把整個 Python 環境或 `.venv` 放到 Google Drive。Colab 的系統環境可能變動，整包快取反而比較容易壞。
- Web UI 可以用 SAM3 或模板偵測。SAM3 不需要水印模板；模板模式需要上傳或放入模板圖片。
- Web UI 進階參數按「保存設定」後會寫到後端設定檔。啟用 Google Drive 快取時，設定檔會放在 `/content/drive/MyDrive/RemoveWatermark/web_advanced_settings.json`，下次重新啟動 Web UI 會自動讀取。
- 如果第二格略過 SAM3 準備，批量處理格會自動改用模板模式，避免直接進入不可用的 SAM3。
- 如果看到 `ModuleNotFoundError: No module named 'remove_watermark'`，通常表示第 1 格沒有成功取得專案，或第 2 格沒有完整安裝。請先重新執行第 1 格，再重新執行第 2 格。
- 第二格會保留 Colab 目前內建的 NumPy 版本，避免在第一次執行時替換 NumPy 造成二進位套件混用。一般全新工作階段不需要重啟。
- 如果仍看到 `numpy.dtype size changed` 或 `NumPy 二進位套件版本不一致`，通常是某個套件已在同一個 runtime 裡替換過 NumPy。請到 `執行階段 -> 重新啟動工作階段`，再從第 1 格開始重新執行。
- Colab 免費 GPU 可能會中斷或被回收，處理很多圖片時建議分批。
- Web UI 的結果會先暫存在工作區。請用 Web UI 上方的「下載目前圖片」或「下載全部圖片」取回正式成品。
- 如果使用最後一格批量命令列模式，結果才會直接寫到 Colab 的 `/content/RemoveWatermark/output`。如果要長期保存，請下載或複製到 Google Drive。
- 網頁介面在 Colab 裡是透過 Colab 的連接埠轉接功能開啟，不是一般本機的 `127.0.0.1`。

## 取回結果

使用 Web UI 時，最簡單的方式是在網頁上按「下載目前圖片」或「下載全部圖片」。

如果你使用最後一格批量命令列模式，才需要在 Colab 左側檔案面板打開：

```text
/content/RemoveWatermark/output
```

接著對需要的圖片或資料夾按右鍵下載。

如果要存到 Google Drive，可以在筆記本新增一格執行：

```python
from google.colab import drive
drive.mount("/content/drive")
!cp -r /content/RemoveWatermark/output /content/drive/MyDrive/RemoveWatermark-output
```
