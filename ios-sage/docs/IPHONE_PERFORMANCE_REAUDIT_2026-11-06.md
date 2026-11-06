# Sage A1.5 原生 iPhone 推理：效能與準確度重新審計（2026-11-06）

## 結論

1 tok/s 只係 Swiftlet 在 iPhone 17 的特定版本和模型配置的開發者報告，並非 iPhone 35B MoE 上限。另一個 Flash-iOS 開發者文檔報告同類 Qwen3.5-35B-A3B 於 iPhone 17 為 5.5 tok/s，Better Stack 2026-09-17 示範報告約 11 tok/s。後兩者沒有經過本專案相同工作負載的獨立實機重測，且 Better Stack 影片稱 iPhone 17、2026-09-21 文章又稱 iPhone 15 Pro，裝置欄位存在衝突。**不能把任何一個數字直接當作 Sage A1.5 Q2_K 實測。**

### 可信度等級

| 發現 | 類別 | 證據性質 |
|---|---|---|
| Sage 基礎：40 decoder layers、256 experts/layer、top-8、width 512、hidden 2048 | 保留、核實 | InternScience model config、GGUF header、模型家族 spec |
| Swiftlet iPhone 17 約 1 tok/s、2.5GB RAM | 保留但限定 | Swiftlet 作者報告，特定 4bit Qwen3.6 |
| Flash-iOS iPhone 17 約 5.5 tok/s | 新增 | 工程 REVIEW 文檔報告，Qwen3.5 35B；沒有附有獨立同條件執行 trace |
| Better Stack 約 11 tok/s | 新增 | 2026-09 的影片示範與文章，量化/設備描述有歧異 |
| Routide 真機 flash paging 成功、分配的快取下 hit rates 變化大 | 保留、補充 | 2026-09-24 預印本、記錄的實驗與程式碼 |
| 目前 Sage Q2_K iPhone 速度 | 未知 | 無正式 tokenizer→router→Metal→token benchmark |

## 重要算術修正：Flash-iOS REVIEW 混用了不同 expert 尺寸

對於 Sage/Qwen3.5 35B，單個 routed expert = gate/up/down 三個矩陣 × hidden 2048 × intermediate 512 = **3,145,728 params**。

每個 token 需要 40 層 × 每層 8 個 routed experts = **320 experts**。

按 GGML K 量化塊字節計算（不包含讀取系統開銷、其他 dense tensors、OS page faults）：

| 全部 routed experts 都是指定量化 | bytes / 256 weights | 每 expert (MiB) | 每 token 全冷讀取 (MiB) |
|---|---:|---:|---:|
| Q2_K | 84 | 0.9844 | **315** |
| Q3_K | 110 | 1.2891 | **412.5** |
| Q4_K | 144 | 1.6875 | **540** |

ANEMLL `FlashMoE-iOS/REVIEW.md` 用「約 7MB/expert × 8 × 40 = 2.24GB/token」推導 iPhone page-cache 命中率。但 35B/Q4_K 的 expert 只有約 1.77 *十進制* MB，不是 7MB。7MB 大小相當於隱藏維度和中間維度都更大的 397B 類模型（單 expert 4 倍參數）。因此該文檔的「由 5.5 tok/s 推算 75%+ cache hit」不能作為 35B 的有效量測結論。**這個審計不否定其自述 5.5 tok/s 的觀察，只否定錯誤尺寸衍生的估算。**

## 簡化 I/O 屋頂模型（非手機實測）

令每 token 實際 cold expert bytes = full-cold bytes × (1 − **按 bytes 加權** cache hit ratio)；其理想化 I/O 上限 = 有效實體讀取吞吐量 ÷ cold bytes。

假設 1.5–3.0 GB/s *有效* SSD 讀取量（僅作情境參數，**並非實測此裝置**）：

| Q2_K 情境 | 每 token cold bytes | 僅 I/O 理論上限 |
|---|---:|---:|
| 全部 cache miss | 330.3 MB | 4.5–9.1 tok/s |
| 50% bytes cache hit | 165.2 MB | 9.1–18.2 tok/s |
| 75% bytes cache hit | 82.6 MB | 18.2–36.3 tok/s |

上表**不是**最終 FPS 估計；Metal GEMV、DeltaNet、GQA、router、LM head、逐層同步、數百次 pread/syscall、電源溫度，會令實際速度更低。cache hits 指向 OS page cache 或明確管理的可用權重，不能將 OS page cache 和 app LFU cache 算成兩份獨立 RAM。

## Source-to-source 核驗

- Swiftlet 4bit 35B iPhone 17 ~1 tok/s；報告 Mac M5 7–11 tok/s，不能套入 iPhone。https://github.com/leonickson1/Swiftlet
- ANEMLL Flash-iOS engineering review 5.5 tok/s on iPhone 17，但其 2.24GB/token 推論不適用 35B。https://github.com/Anemll/Flash-iOS/blob/main/FlashMoE-iOS/REVIEW.md
- Better Stack 2026-09-21 文章 11 tok/s、tiered 2/4bit 約13.4GB；影片 2026-09-17 明確說 iPhone 17、文章說 15 Pro。https://betterstack.com/community/guides/ai/35b-model-on-iphone/ ；https://www.youtube.com/watch?v=hM_mEE8BXfU
- Routide 對 Qwen3.6-35B-A3B 4bit 真機測試：512MiB LRU 0% route replay hits，576MiB 38.58%，說明 cache policy 甚至比增大少量容量更重要。實機 128 output 的請求 wall time 約 63–89 s（含 prompt prefill，不等於純 decode）。https://arxiv.org/abs/2609.29032
- PowerInfer-2 2024 論文在**不同手機及 TurboSparse Mixtral 47B** 上報告 11.68 tok/s，僅證明 smartphone flash-aware MoE > 10 tok/s 在某種情況可能，不能當作 Apple/Sage 速度。https://arxiv.org/abs/2406.06282
- Sage Q2_K 13.2GB / Q3 15.5+GB / Q4_K_M 21.7GB 的已發佈檔案清單。https://huggingface.co/mradermacher/Sage-A1.5-GGUF
- QuantDiff 在 500 題數據：i1-Q2_K 434/500 (86.8%)、Q4_K_M 449/500 (89.8%)；tool call Q2 164/200 (82%) vs Q4 170/200 (85%)，信賴區間重疊，不能宣稱相同品質，也不能斷言退化有統計顯著性。未進行 BF16 KLD。https://quantdiff.com/models/sage-a1.5/
- na0x2c6 模型卡提供 IQ3_S-exps-down4 / Q4_K imatrix KLD 對比，顯示有精度分層保護 down-experts 的研究價值。https://huggingface.co/na0x2c6/Sage-A1.5-GGUF

## 新方向：比從零移植 Strata 更務實

先以已存在的 **Flash-iOS C/Objective-C/Metal Qwen3.5 35B 全圖推理** 作為母版（而非把 Windows CUDA binaries 移植到 iPhone）；將 **Sage 官方權重**、Sage tokenizer/template、MTP 關聯、Q2/Q4 混合量化、固定 stride 的 expert files 接入。同一實例應保留全部 8 selected experts + 1 shared，絕不可為了快而靜默跳過冷專家；驗證 logits parity、路由結果、工具 JSON、長文與測試題，再優化。

實驗變數依序：
1. Q4 35B baseline 對 Sage Q4 文件作 tokenizer / attention / router correctness；確認完整推理後再升級 Q2_K / mixed experts。
2. 先依靠 OS page cache + bounded staging buffers，和自建 LFU/recency cache 做 A/B。現有 branch 的 Swift pager 只是一個 raw tensor storage prototype，未接 Metal；自建 cache 並不天然比 OS 快。
3. 用 per-layer expert usage route logs 在多樣任務（中英文、coding、reasoning、tool calling）選 hot experts 4-bit、cold experts 2-bit；down projection 優先保精度；唔好只按單一 prompt 量化。
4. post-route parallel pread, page-aligned chunk fanout、CPU/GPU overlap、避免不必要的大型 memcpy、batch prefill。
5. 待基本 logits parity 正確才可實驗 MTP，必須量度 draft acceptance、真實 wall time、功耗和錯誤率。

### 對現有 iOS pager 原型新增的效能審核

目前 `ios-sage/Sources/SagePagedExperts/Pager.swift` 使用原生 GGUF 中分開的 gate、down、up tensor 段，對每一個 selected expert 進行 **3 次 pread**；對 40 層 × 8 路由，即 **960 次 pread 呼叫 / token**（全部 miss 的情況），加上逐層同步和 CPU 串行處理。這是功能驗證用的索引器，不是經過性能調校的推理權重格式。

**P0 優化：** 於安裝／離線轉換階段，將一個 expert 的 gate/up/down 量化塊與描述資料集中在固定步長、page-aligned 的 compact blob；運行時每 expert 一次讀取，降到理論 **320 次 pread / token**，同層 8 expert 讀取可批次併行。不得更改量化數值語義或跳過 expert；讀取、dequant、logit parity 和 SHA-256 是必要驗證。

因 iOS 已有 OS page cache，盲目增加 app LFU 快取可能把權重複製兩次、加劇 jetsam。須比較 page-cache-only / app-LFU / hybrid hot-pin 三組，且每組記錄 process footprint、真正讀取 byte、token latency；現有 Swift pager 不應作為最終執行時的預設最快方案。

### 驗收方法

**同一 iPhone**、同一 thermal state、Release build、每種設定冷/暖各≥3 次，記錄：
- 精確裝置型號、iOS build、RAM、剩餘容量、檔案 SHA-256、Git commit、量化格式、權重大小。
- prefill prompt 32 / 128 / 512 / 2048 tokens；decode 128 / 512 tokens；context 2048 / 8192。
- TTFT、純 decode tok/s、請求端到端 tok/s、P50/P95 token latency、page-cache/miss proxy、actual expert bytes read。
- process footprint + Metal allocated size + iOS jetsam + thermal state + time; battery/energy instrumentation scope。
- token/route/logits compare vs host baseline；不漏任何 routed expert；同品質任務對比。
- **驗收目標（非預測）**：首先使正確 Sage Q2/mixed ≥3 tok/s 持續、再試 ≥5；將 8–12 tok/s 當 stretch goal，不承諾。若只有峰值 11、持續 3，要如實報告。

## 當前專案狀態

本分支已實作 GGUF expert offset 索引及 Swift pread/LFU byte-budgeted expert pager，**只通過 synthetic/unit-level checks**；未有 Sage Q2_K Metal GEMV、完整 tokenizer→logits graph、iPhone 實機成功輸出。絕不能把原型標示為 3/5/11 tok/s。

本審計旨在保留前面可驗證成果，修正此前慢速上限判斷與外部文檔尺寸錯誤，新增更高速技術參照與可重複的下一步驗證方法。
