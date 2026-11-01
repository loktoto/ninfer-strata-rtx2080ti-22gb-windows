# Sage A1.5 完整設定：Windows / RTX 2080 Ti 22 GB

本文件日期：2026-11-01。Sage 是模型，GGUF 是模型容器，Strata 是推論引擎。目前 D 配置在 Strata 內使用移植自 NInfer-all 的 IQ4_XS expert kernel，再配合 Strata MTP；不是同時啟動兩套完整引擎。獨立 NInfer 的 C 配置仍未完成全模型推論。

## 1. 這台已準備好的 PC

不用重裝、重新 clone 或覆寫已測 build。開啟 PowerShell：

~~~powershell
Set-Location D:\AI\ninfer-strata-test
$sageModel = 'D:\AI\LLM\models\sage\Sage-A1.5-IQ4_XS-exps.gguf'
Test-Path $sageModel
nvidia-smi
.\tools\start_iq4xs.ps1 -Model $sageModel -DryRun
.\tools\start_iq4xs.ps1 -Model $sageModel
~~~

Test-Path 應為 True。DryRun 只核對路徑和列出設定，不載入 GPU、不提取 tokenizer、不驗證模型。正式啟動會提取 GGUF 內的 tokenizer/chat template，在新的 sage-api-* 目錄保存 strata-sage.json，然後載入模型。等待 ready，保持該視窗開啟，按 Ctrl+C 停止。

本機已測 GGUF 為 19,322,710,880 bytes，SHA256：
be92beb4e0fda7b40fc4dd2705831095c27792d342833e9d06d9366850bf7069。
需要確認身分時可執行 Get-FileHash -Algorithm SHA256 -LiteralPath $sageModel；讀完整個大檔會花時間。

## 2. 預設與原理

| 項目 | 預設 | 意義 |
| --- | --- | --- |
| Context | 32768 | 歷史、模板、輸入、預留輸出共用容量 |
| KV cache | FP16 | attention 的 K/V 暫存；不是模型權重量化 |
| CUDA graphs | 開 | 重播 GPU 工作，減少 CPU launch 開銷 |
| NInfer IQ4_XS expert kernel | 開 | 替換指定 expert 計算 |
| MTP window | 4 | 一個已知 token，加最多三個猜測 |
| CPU threads | 16 | CPU 工作執行緒，不是 GPU cores |
| Lookup speculation | 0 | 本配置不用重複文字 lookup 草稿 |
| 跳過 layers/experts | 關 | 不靠省略模型工作加速 |
| 舊 RMSNorm candidate | 關 | 有未解決的 logits 差異 |
| Native fusion | 關 | 可選，但有 prefill/延遲退步 |
| Packed Q6 head | 關 | 沒有一致收益，增加約 398 MiB VRAM |

### Context 與 max_tokens

Context 單位是 token，不是中文字數。系統提示、歷史、模板標記、輸入和輸出全部佔空間。大致需要「格式化後輸入 tokens + max_tokens ≤ context」，還應保留內部處理空間。API 的 max_tokens 是單次輸出上限，不能擴大 context。

目前 fit_max_tokens=false、drop_oldest=false；不要假設伺服器會自動縮短輸出或刪掉最舊對話。遇到超限應清理/摘要歷史或減少輸出要求。

上游 KV 可隨對話增長，並與 expert cache 共用 VRAM；32K configured 不等於啟動便配置全部 32K KV。真正佔用 context 增加仍可能改變 VRAM、expert residency 和速度。現有速度測試提示為 42/52/1133 tokens，不能當成已通過 32K 滿載，更不能宣稱 128K/256K 已驗證。

~~~powershell
# 明確指定預設配置
.\tools\start_iq4xs.ps1 -Model $sageModel -MaxContext 32768 -Threads 16 -DraftWindow 4

# 記憶體問題診斷；不是新的已驗證效能配置
.\tools\start_iq4xs.ps1 -Model $sageModel -MaxContext 8192
~~~

每次執行另一個啟動命令之前，先停止原服務。上游雖有 Q8 KV 選項，但改變數值且未通過本組比較；簡易啟動器固定 FP16。不可用不同精度的 D 和 FP16 A 混稱同條件兩倍加速。

### MTP 與 window

MTP 猜下一批 token，由主模型一起驗證；接受的猜測可減少逐 token 執行次數。猜測本身和驗證都有成本，猜錯要修正，因此 window 越大不保證越快。不是四份回答，也不是保證四倍速度。

Wrapper 支援 window 2、3、4，最多各猜 1、2、3 個 token。原始 block draft 上限是三個猜測；本配置 lookup=0，把 raw window 設為 8 不會自動猜七個。實際驗證批量會由策略調整。相容 draft block 已在這個 GGUF 內，不需另下載 draft model。此 wrapper 的 MTP 需要 CUDA graphs。

~~~powershell
.\tools\start_iq4xs.ps1 -Model $sageModel -DraftWindow 2
.\tools\start_iq4xs.ps1 -Model $sageModel -DraftWindow 3
.\tools\start_iq4xs.ps1 -Model $sageModel -DraftWindow 4
.\tools\start_iq4xs.ps1 -Model $sageModel -NoDraft
~~~

這些是替代選項，不應全部同時執行。NoDraft 優先，DraftWindow 此時不生效。Window 4 是目前短工作負載測得較好，不代表長文/其他採樣永遠最佳。看 accepted/offered、decode tok/s、首 token 延遲、總耗時和正確性，不能只看接受率。

## 3. 驗證 API 與接駁客戶端

保留 server 視窗，另開 PowerShell：

~~~powershell
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod http://127.0.0.1:8080/v1/models
$payload = @{
    model = 'sage-a1.5'
    messages = @(@{ role = 'user'; content = 'Explain what a prime number is.' })
    temperature = 0
    top_p = 1
    max_tokens = 256
    seed = 1
    cache_prompt = $false
    chat_template_kwargs = @{ enable_thinking = $false }
    stream = $false
} | ConvertTo-Json -Depth 8
$answer = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8080/v1/chat/completions' -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($payload))
$answer.choices[0].message
$answer.usage
~~~

預設無 API key。若自行設定 STRATA_API_KEY，請求需帶相同 Bearer token。Thinking 開關要配合模型模板；對比性能時保持相同推理設定，推理文字及輸出長度差異也會影響總工作量。

聊天客戶端選 OpenAI-compatible：
- Base URL：http://127.0.0.1:8080/v1
- Model：sage-a1.5
- API key：無金鑰時按客戶端要求使用空白或 placeholder；有金鑰時兩邊相同。
- 先用 temperature 0、max output 256 測通，再自行調整。

目前只綁本機 loopback。另一部電腦不能用它自己的 127.0.0.1 連這台主機。這份配置不代表已完成遠端或多人併發部署。

## 4. 比較配置與保存

~~~powershell
# D：combined + MTP
.\tools\start_iq4xs.ps1 -Model $sageModel
# Strata native expert + MTP
.\tools\start_iq4xs.ps1 -Model $sageModel -Native
# combined，關閉 MTP
.\tools\start_iq4xs.ps1 -Model $sageModel -NoDraft
# Strata native，關閉 MTP
.\tools\start_iq4xs.ps1 -Model $sageModel -Native -NoDraft
# 可選 fusion，需要已有 build-fused
.\tools\start_iq4xs.ps1 -Model $sageModel -FusedNormQ8
~~~

Native 仍是 Strata，並非 A 的 llama.cpp 或 C 的 NInfer。移除 FusedNormQ8 回到預設 build-optimized；該開關不會自動編譯。自訂引擎使用 -Engine 完整路徑，必須支援所選功能。

每次新啟動會保存設定及 tokenizer，請保留 sage-api-* 和引擎/模型 hash。移動檔案會令已保存絕對路徑失效。完整 saved-config 重啟流程見 [英文設定指南](IQ4XS_SETUP.md#restart-a-saved-configuration)。

## 5. 新電腦／新資料夾從零建立

已準備好的 PC 不需要重做。

1. 準備 Git、Python（本次主專案為 3.12）、Visual Studio 2022 Build Tools 的 C++ desktop workload / Windows SDK、相容 NVIDIA driver 和 CUDA toolkit。已測 CUDA 13.0.88、MSVC 19.44、SM75；其他版本需重驗證。
2. 使用 x64 Developer PowerShell，確認 git、python、cl、nvcc 能執行。只有 runtime DLL 不能編譯，還需要 compiler 和 headers。
3. Clone 至新資料夾；私有 repository 需要存取權：

~~~powershell
Set-Location D:\AI
git clone https://github.com/loktoto/ninfer-strata-gpu-unlock-rtx2080ti-22gb-windows.git sage-clean
Set-Location .\sage-clean
python -m venv .venv
$env:PATH = "$(Resolve-Path '.\.venv\Scripts');$env:PATH"
python -m pip install cmake ninja
python tools\bootstrap.py
python -m pip install -r .deps\PolyStrata\requirements.txt
.\tools\build_windows.ps1 -BuildDirectory build-optimized -Jobs 2
~~~

每步成功才繼續。Bootstrap 固定 upstream 與 patches，遇到分歧會拒絕覆寫。Jobs 2 是編譯併行，不是推論 Threads。不同 GPU 架構不能當成本 SM75 配置已驗證。

4. 另行取得完整相容 IQ4_XS GGUF；repository 不含模型。原本 Q4_K_M model lock 是另一檔案，其 hash 不可代替本模型。
5. 確認 CUDA runtime DLL 在 PATH；prepared PC 的 .cuda/bin/x64 是本地準備內容，fresh clone 不自帶。使用實際 toolkit 的 runtime 路徑。
6. 執行第 1 節 DryRun、啟動和第 3 節 API 測試。新 build 不自動繼承歷史測量資格，正式比較仍需數值/模型驗證。獨立 NInfer 實驗另用 Python 3.11，啟動 D 不需要它。
7. 如需 fusion，另執行 build_windows.ps1 -BuildDirectory build-fused -Jobs 2 並驗證，才使用 FusedNormQ8。

## 6. 排錯與公平測試

- Missing engine/Python/DLL：檢查 .venv、build-optimized/strata/strata-poly.exe 和 CUDA PATH。
- PowerShell policy 阻擋：在允許情況可對單次程序用 powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\start_iq4xs.ps1 -Model $sageModel，不必改永久政策。
- Port 8080 被佔：Get-NetTCPConnection -LocalPort 8080 -State Listen；識別自己的舊服務後回原視窗 Ctrl+C。
- OOM：檢查 nvidia-smi，停止自己不需要的 GPU 工作，以 8192 context 診斷。22 GB 並不保證任何長度。
- 回答被截斷：檢查 finish reason、max_tokens、context 及 engine logs。
- Tok/s 高但體感慢：看 prefill、TTFT、wall time；長提示/短回答容易由 prefill 主導。
- 同條件比較需固定 GGUF、KV、提示、context、採樣、輸出上限及 cache。一次只跑一個引擎，先暖機，再交錯次序多次啟動，保留所有樣本和時脈/溫度。
- 32K configured 與 32K occupied 是不同測試；長文驗證要量實際 tokens、VRAM、品質、錯誤及速度。
- [100% uplift 重新審視](PERFORMANCE_2X_REVIEW_2026-11-01.md)：目前未證明兩倍，不用跨日混合數字或不同精度湊成目標。

