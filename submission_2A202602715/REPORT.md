# Báo cáo Lab Day 1 — MSSV 2A202602715

Mọi con số dưới đây lấy từ `experiments.xlsx` (một dòng cho mỗi `exp_id`) và `eval_result.json`. Mọi so sánh đều làm trên **val**; eval chỉ dùng ở mục 4.

## 1. Thiết lập

- **Môi trường:** Google Colab, GPU Tesla T4, PyTorch 2.11.0+cu130. Kết quả ghi thẳng vào Google Drive để không mất khi Colab ngắt kết nối.
- **Dữ liệu:** Forest CoverType; `train` 464 809 / `eval` 116 203 theo `split_metadata.csv`. Validation lấy 20% của train (phân tầng, seed 42), còn 371 847 mẫu train và 92 962 mẫu val. Chuẩn hoá 10 cột số bằng mean/std của phần train còn lại.
- **Model:** `M-base` (54→256→128→7, 47 879 tham số, có `assert`), ReLU, không softmax trong model.
- **Baseline:** CE, SGD+momentum 0,9, lr = 0,3 (chọn bằng val, mục 2), batch 512, 20 epoch, khởi tạo He (`kaiming_normal_`, bias = 0), không dropout, không clip, FP32.
- **Mốc tham chiếu:** "đoán lớp đa số" cho accuracy 0,4876 trên val.
- **Chủ đề đã thử:** ☑ loss ☑ optimizer ☑ hyper-parameter ☑ dropout ☑ clipping ☑ mixed precision ☑ init (đủ 7/7). Tổng cộng 39 lần chạy.
- **Đo đạc:** train loss đo ở chế độ `eval()` trên một tập con cố định 50 000 mẫu train. `grad_norm` là chuẩn L2 toàn cục đo **trước** khi clip, ở mỗi bước. Metric báo cáo lấy ở best epoch (val loss thấp nhất).

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (B, 7) |
| Loss bước 0 (ln 7 = 1,946) | 2,269 (He, seed 1); 1,946 với init `normal` |
| Quá khớp 20 mẫu: loss cuối | 6,6·10⁻⁶ sau 300 bước Adam, accuracy 100% |
| Mọi tham số có gradient khác 0 | ☑ có (‖grad‖ từ 0,27 tới 1,94) |
| Baseline, số seed | 3 (`base-s1`, `base-s2`, `base-s3`) |
| Baseline: val acc (TB ± σ) | 0,9120 ± 0,0010 |
| Baseline: val macro-F1 (TB ± σ) | 0,8580 ± 0,0037 |

**Ngưỡng nhiễu dùng trong báo cáo:** 2σ = **0,0074** (val macro-F1).

- **Loss bước 0 cao hơn ln 7 khoảng 0,32.** He (Var = 2/n_in) được áp cả cho lớp ra, nên logit ban đầu có std ≈ 0,58 thay vì ≈ 0 và softmax không còn đều. Với init `normal` 0,01, logit ≈ 0 và loss bước 0 = 1,9460, đúng ln 7. Điều đó cho thấy cách tính loss không sai; mức lệch là do thang của lớp cuối.
- **Hình dạng đường cong baseline** (`figures/base-s1.png`): val loss giảm đều từ 0,445 xuống 0,222, best epoch 19–20. Khoảng cách val − train loss chỉ 0,027 và cả hai còn giảm. Mô hình **chưa hội tụ và chưa quá khớp**.

## 3. Kết quả theo chủ đề

Δ ghi trong mục này là chênh lệch val macro-F1 so với trung bình baseline (0,8580). Ngoài baseline, các thí nghiệm chỉ chạy 1 seed (seed 1).

### Dò lr cho baseline (SGD+momentum)

| lr | 0,01 | 0,03 | 0,1 | 0,3 |
|---|---|---|---|---|
| val macro-F1 | 0,7615 | 0,8240 | 0,8410 | **0,8599** |

lr = 0,3 tốt nhất nhưng nằm ở **biên lưới**, nên có thể lr lớn hơn còn tốt hơn; đây là một hạn chế. Ảnh: `figures/compare_baseline_lr.png`.

### 3.1 Hàm mất mát — CE vs MSE

- **Dự đoán:** CE có ∂L/∂z = softmax − y, nên gradient vẫn lớn khi dự đoán sai nặng. MSE trên logit (trung bình trên B×7 phần tử, không có hệ số 1/2) cho gradient nhỏ hơn, nên học chậm hơn.
- **Kết quả:**
  - `loss-mse`: val macro-F1 0,7649 (Δ = −0,093 ≫ 2σ), val acc 0,8801 so với 0,9120.
  - `loss-mse-lr3x` (lr 0,9): còn kém hơn, 0,7317. Hai lần chạy này được so bằng metric, không so loss.
  - Ảnh: `figures/compare_loss.png`.
- **Giải thích:**
  - grad_norm trung vị của MSE là 0,065, của CE là 0,346, tức nhỏ hơn ~5 lần. Gradient MSE theo logit là 2(z − y)/7, tăng tuyến tính chứ không bão hoà về hướng đúng như CE. Hơn nữa khi z đã lớn, MSE còn phạt cả việc logit đúng "quá tự tin".
  - **Không như dự đoán:** tăng lr ×3 không bù được mà còn tệ hơn; grad_norm max tăng lên 11,8. Vậy nguyên nhân không chỉ là thang gradient. Với lr 0,9 và momentum 0,9, bài toán hồi quy logit dao động mạnh.

### 3.2 Bộ tối ưu hoá

- **Dự đoán:** SGD thuần cần lr lớn hơn ~10 lần so với SGD+momentum. Adam/AdamW nhanh ở vài epoch đầu. Adam và AdamW (wd 0,01) gần như bằng nhau.

| Bộ tối ưu | lr thử | lr tốt nhất (exp_id) | val macro-F1 | best epoch | Δ | vượt 2σ? |
|---|---|---|---|---|---|---|
| SGD | 0,03 / 0,1 / 0,3 | `opt-sgd-lr0.3` | 0,8090 | 18 | −0,049 | có (kém hơn) |
| SGD+momentum 0,9 | 0,01 / 0,03 / 0,1 / 0,3 | `opt-sgdm-lr0.3` | 0,8599 | 20 | +0,002 | không |
| Adam (0,9; 0,999; 1e-8) | 3e-4 / 1e-3 / 3e-3 | `opt-adam-lr0.003` | **0,8660** | 18 | +0,008 | vừa vượt |
| AdamW (wd 0,01) | 3e-4 / 1e-3 / 3e-3 | `opt-adamw-lr0.003` | 0,8615 | 19 | +0,0035 | không |

- **Độ nhạy với lr** (`figures/compare_optimizer_lr.png`): cả bốn bộ đều đạt tốt nhất ở **lr lớn nhất của lưới**. Đi xuống một nấc lr (÷3), SGD mất 0,056, Adam mất 0,018 và SGD+momentum mất 0,019. Ở lr nhỏ nhất, Adam 3e-4 (0,787) và SGD+momentum 0,01 (0,762) đều chậm rõ.
- **Giải thích:**
  - SGD thuần ở lr 0,3 (0,809) tương đương SGD+momentum ở lr 0,03 (0,824). Lý do: momentum μ = 0,9 khuếch đại bước khoảng 1/(1−μ) = 10 lần.
  - Adam chia bước cho √v̂ của từng tham số nên ít phụ thuộc thang gradient. Ảnh `figures/compare_optimizer.png` cho thấy val loss của Adam xuống nhanh ở các epoch đầu.
  - Adam chỉ hơn baseline 0,008, vừa vượt 2σ = 0,0074, và mới có 1 seed. Đây là bằng chứng yếu.
  - AdamW và Adam chênh 0,0045 < 2σ. Với wd 0,01 và mô hình chưa quá khớp, suy giảm trọng số gần như không có tác dụng.

### 3.3 Hyper-parameter

| exp_id | Thay đổi | bước/epoch | s/epoch | val macro-F1 | Δ |
|---|---|---|---|---|---|
| `hp-batch128` | batch 128 | 2 906 | 4,79 | 0,8035 | −0,054 |
| `hp-batch2048` | batch 2048 | 182 | 0,32 | 0,8420 | −0,016 |
| `hp-batch2048-lr4x` | batch 2048, lr 1,2 (đổi 2 yếu tố) | 182 | 0,29 | 0,0937 | sụp |
| `hp-wide` | M-wide 512-256 | 727 | 1,23 | **0,8721** | **+0,014** |
| `hp-deep` | M-deep 256-128-64 | 727 | 1,36 | 0,8588 | +0,001 |

- **Batch 128 (khác dự đoán).** Có gấp 4 số bước nhưng kém hơn rõ: best epoch 15, sau đó val loss tăng. Ở lr 0,3 với lô nhỏ, phương sai gradient lớn gấp ~4 lần (tỉ lệ η/B tăng ×4), nên quỹ đạo nhiễu và không hội tụ tốt. Mỗi epoch cũng chậm gấp 4 do phải gọi nhiều kernel nhỏ hơn.
- **Batch 2048 cùng lr.** Ít bước cập nhật hơn 4 lần nên học chậm hơn, đúng dự đoán.
- **Quy tắc tăng lr theo lô (không như dự đoán).** Áp dụng mà không warmup thì mô hình **sụp**: val loss đứng ở 1,205, bằng entropy của phân bố lớp; acc 0,4876; F1 0,094, tức luôn đoán lớp đa số. Quy tắc trong slide đi kèm khởi động lr, và thí nghiệm này cho thấy vì sao cần nó.
- **M-wide và M-deep.** M-wide vượt 2σ: mô hình đang chưa khớp, nên thêm năng lực giúp ích (gap val − train vẫn chỉ 0,030). M-deep không khác baseline.

### 3.4 Dropout

| exp_id | q | best val loss | final train loss | gap val − train | val macro-F1 | Δ |
|---|---|---|---|---|---|---|
| `base-s1` | 0 | 0,2221 | 0,1950 | 0,027 | 0,8599 | — |
| `drop-0.1` | 0,1 | 0,2398 | 0,2248 | 0,015 | 0,8339 | −0,024 |
| `drop-0.3` | 0,3 | 0,3165 | 0,3107 | 0,006 | 0,7820 | −0,076 |
| `drop-0.5` | 0,5 | 0,4198 | 0,4165 | 0,003 | 0,6698 | −0,188 |

- Gap thu hẹp khi q tăng, nhưng là vì **cả train lẫn val loss đều tăng**, không phải vì val giảm (`figures/compare_dropout.png`).
- Mô hình **không quá khớp**: gap baseline chỉ 0,027 và val loss vẫn đang giảm. Dropout lúc này chỉ bớt năng lực và thêm nhiễu cho gradient, nên kết quả tệ hơn, đúng dự đoán. q được chọn dựa trên mức quá khớp: gap gần 0 nghĩa là không nên dùng dropout.

### 3.5 Gradient clipping

- **Chọn c:** grad_norm theo bước của `base-s1` có p50 = 0,346, p90 = 0,405, max = 2,84 (gai ở các bước đầu). Lấy **c = 0,346** để clipping thực sự kích hoạt.
- **Ở lr thường** (`clip-0.346`): clipping kích hoạt ở 72% số bước. Val macro-F1 0,8469 (Δ = −0,011, vượt 2σ theo chiều xấu). Cắt ở trung vị gần như là giảm lr hiệu dụng, trong khi baseline vốn đang thiếu bước.
- **Ở lr cao** (lr ×10 = 3; `figures/compare_clipping.png`):
  - Không clip (`clip-none-highlr`): grad_norm có **gai 7 814** ở epoch 1. Sau đó mạng sụp về dự đoán hằng: val loss 1,21 ≈ entropy lớp, F1 0,094, grad_norm chỉ còn ~0,27. Loss không thành NaN nên cờ `diverged` = N, nhưng huấn luyện đã hỏng.
  - Có clip (`clip-0.346-highlr`): grad_norm max chỉ 3,7, mạng **không sụp** (F1 0,26 ở epoch 1, ~0,19 về sau). Clip kích hoạt ở 43% số bước của epoch 1, 9% ở epoch 2–3, rồi ≤ 2,5% từ epoch 4. Tuy vậy kết quả vẫn rất kém, vì mỗi bước vẫn dài tới lr·c ≈ 1 và momentum còn cộng dồn thêm.
- **Kết luận:** clipping chặn được gai gradient đột ngột, giữ cho một bước xấu không phá hỏng cả mạng, nhưng không thay được việc chọn lr đúng.
- **Phỏng đoán chưa đo:** cơ chế sụp có lẽ là ReLU chết sau bước khổng lồ (bias/trọng số bị đẩy âm). Tôi không đo tỉ lệ nơ-ron chết, nên đây chỉ là giả thuyết, phù hợp với việc chỉ còn bias lớp ra học được phân bố lớp.

### 3.6 Mixed precision

| exp_id | precision | s/epoch | peak MB | val macro-F1 | Δ |
|---|---|---|---|---|---|
| `base-s1` | FP32 | 1,21 | 173 | 0,8599 | +0,002 |
| `amp-fp16` | FP16 + GradScaler | 1,72 | 173 | 0,8552 | −0,003 |
| `amp-bf16` | BF16 | 1,45 | 173 | 0,8506 | −0,0074 |
| `hp-wide` | FP32, M-wide | 1,23 | 190 | 0,8721 | +0,014 |
| `amp-fp16-wide` | FP16, M-wide | 1,67 | 190 | 0,8685 | +0,011 |

- **Không nhanh hơn, mà chậm hơn: FP16 +42%, BF16 +20%** (đúng dự đoán). Mạng có 48k–161k tham số với batch 512, nên mỗi matmul quá nhỏ để Tensor Core có lợi. Thời gian bị chi phối bởi chi phí cố định mỗi bước: gọi kernel, các phép cast của autocast, và với FP16 thêm `unscale_` cùng kiểm tra inf của GradScaler.
- **Bộ nhớ như nhau,** vì phần lớn bộ nhớ đỉnh là dữ liệu FP32 nằm sẵn trên GPU.
- **Độ chính xác nằm trong nhiễu.** BF16 có |Δ| = 0,0074, sát ngưỡng, nên chưa kết luận được là kém hơn; BF16 chỉ có 8 bit mantissa.
- **T4 và BF16:** T4 (Turing) không có phần cứng BF16; PyTorch vẫn chạy được nhưng chậm.
- **Vì sao FP16 cần GradScaler còn BF16 thường không:** FP16 có khoảng biểu diễn hẹp (max 65 504, min normal ≈ 6·10⁻⁵), nên gradient nhỏ dễ underflow về 0 và cần nhân loss với s. BF16 có cùng 8 bit số mũ như FP32 (≈ 10⁻³⁸ đến 3·10³⁸), nên thường không cần.

### 3.7 Khởi tạo tham số

| init | std ReLU1 | std ReLU2 | std logits | loss bước 0 | val macro-F1 (exp_id) | Δ |
|---|---|---|---|---|---|---|
| zeros | 0 | 0 | 0 | 1,9459 | 0,0937 (`init-zeros`) | sụp |
| normal 0,01 | 0,0203 | 0,0022 | 0,0003 | 1,9460 | 0,8558 (`init-normal`) | −0,002 |
| xavier (`xavier_normal_`, 2/(n_in+n_out)) | 0,1628 | 0,1248 | 0,1916 | 2,0222 | 0,8605 (`init-xavier`) | +0,0025 |
| default (`nn.Linear`) | 0,1603 | 0,0678 | 0,0585 | 1,9830 | 0,8525 (`init-default`) | −0,0055 |
| he (baseline) | 0,3901 | 0,3661 | 0,5773 | 2,2691 | 0,8599 (`base-s1`) | — |

- **`zeros` hỏng (đúng dự đoán).** Kích hoạt ẩn đều bằng ReLU(0) = 0, nên ∂L/∂W = 0 ở mọi lớp và các nơ-ron trong một lớp đối xứng hoàn toàn. Chỉ b3 nhận gradient, nên mô hình học đúng phân bố tiên nghiệm: val loss 1,205 = entropy lớp, acc 0,4876.
- **`normal` 0,01 vẫn học được.** Kích hoạt co ~×10 mỗi lớp, nhưng mạng chỉ có 3 lớp và lr 0,3 đủ lớn để kéo trọng số ra khỏi vùng nhỏ.
- **Mạng 3 lớp không đủ sâu** để thấy khác biệt khi huấn luyện: normal, xavier, default đều nằm trong 2σ so với He.
- **Minh hoạ 30 lớp ReLU** (chỉ forward, trong notebook), std ở lớp 30:
  - normal: → 0 (dưới 10⁻²⁰ từ lớp 22);
  - xavier: 4,6·10⁻⁶;
  - default: 0,021 (phỏng đoán: bias ngẫu nhiên của `nn.Linear` giữ mức này);
  - he: 0,25.
- **He khác Xavier ở đâu:** He có hệ số 2, bù việc ReLU triệt một nửa phương sai, nên phương sai được giữ qua các lớp. Xavier thiết kế cho kích hoạt tuyến tính/tanh nên co dần qua mỗi lớp ReLU. Điều này quan trọng với mạng sâu.

## 4. Đánh giá cuối trên tập eval

**Quy tắc chọn cấu hình** (viết trong notebook trước khi chạy, chỉ dùng val):
1. Lấy bộ tối ưu + lr có val F1 cao nhất: Adam 3e-3.
2. Dùng M-wide/M-deep nếu vượt baseline + 2σ: M-wide vượt.
3. Dùng dropout / batch 128 nếu vượt baseline + 2σ: không cái nào vượt.
4. Huấn luyện 40 epoch, vì baseline chưa hội tụ ở epoch 20; best epoch chọn theo val loss.
5. Chạy 3 seed và nộp seed có val F1 cao nhất.

Lưu ý: cấu hình cuối đổi ba yếu tố cùng lúc so với baseline (optimizer, kiến trúc, số epoch).

| Cấu hình | Seed nộp | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|---|---|---|---|
| Baseline (`base-s1`) | 1 | 0,8599 | **0,8609** | 0,9105 |
| Cấu hình cuối (`final-s3`) | 3 | 0,9105 | **0,9135** | 0,9413 |

- **Cải thiện trên val** qua 3 seed: 0,9079 ± 0,0033 (`final-s1..s3`) so với 0,8580 ± 0,0037, tức +0,050, lớn gấp ~7 lần ngưỡng 2σ.
- **Cải thiện trên eval:** +0,053 (0,9135 so với 0,8609). Eval chỉ chạy cho hai mô hình này, mỗi mô hình 1 seed, nên không có σ cho điểm eval.
- **Val và eval rất gần nhau:** chênh +0,001 với baseline và +0,003 với cấu hình cuối. Val là ước lượng tốt của eval, đúng như ghi chú của giảng viên.

### 4.1 Phân tích lỗi theo lớp (cấu hình cuối, eval)

| Lớp | Loại rừng | support | precision | recall | F1 |
|---|---|---|---|---|---|
| 0 | Spruce/Fir | 42 370 | 0,9427 | 0,9335 | 0,9381 |
| 1 | Lodgepole Pine | 56 660 | 0,9479 | 0,9517 | 0,9498 |
| 2 | Ponderosa Pine | 7 151 | 0,9455 | 0,9357 | 0,9405 |
| 3 | Cottonwood/Willow | 549 | 0,8092 | 0,9271 | 0,8642 |
| 4 | Aspen | 1 899 | 0,8443 | 0,8710 | **0,8574** |
| 5 | Douglas-fir | 3 473 | 0,8927 | 0,9033 | 0,8980 |
| 6 | Krummholz | 4 102 | 0,9374 | 0,9564 | 0,9468 |

- **Lớp khó nhất là lớp 4 (Aspen), F1 = 0,857.** Ma trận nhầm lẫn trong notebook cho thấy 10,3% mẫu Aspen (195/1 899) bị đoán thành lớp 1 (Lodgepole Pine).
  - **Mất cân bằng:** Aspen chỉ có ~6 100 mẫu train, lớp 1 có ~181 000 (≈ 1:30). Ở vùng chồng lấn, biên quyết định nghiêng về lớp đông.
  - **Chồng lấn đặc trưng:** hai lớp cùng dải độ cao (Elevation chuẩn hoá trung bình −0,61 so với −0,14). Aspen khác chủ yếu ở việc gần đường hơn (H_Dist_Road −0,64) và Hillshade_3pm thấp hơn, đều là tín hiệu yếu.
- **Lớp 3 (Cottonwood/Willow):** recall cao (0,93) nhưng precision thấp (0,81). Mô hình gán nhầm 84 mẫu Ponderosa Pine và 36 mẫu Douglas-fir sang lớp 3; ba loài này cùng xuất hiện ở vùng thấp.
- **Hướng cải thiện sẽ thử:** CE có trọng số lớp (∝ 1/√tần suất) hoặc lấy mẫu cân bằng, chọn trọng số bằng val macro-F1.

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào thắng khi chỉnh lr công bằng?**
   - Adam (lr 3e-3) với 0,8660, hơn SGD+momentum (0,8599) và AdamW (0,8615) chưa tới hoặc chỉ vừa bằng 2σ. Kết luận chỉ chắc chắn ở hai điểm: hơn SGD thuần (0,809), và Adam "không thua".
   - Nếu không chỉnh lr thì kết luận phụ thuộc vào lr được chọn. Cùng lr 0,03, SGD đạt 0,658 còn SGD+momentum đạt 0,824. Tôi không chạy SGD ở lr của Adam (1e-3), nên không có số đo cho trường hợp đó.
   - Cả bốn bộ đều tốt nhất ở biên lưới, nên thứ hạng còn có thể đổi nếu mở rộng lưới.
2. **Dropout có giúp khi mô hình chưa quá khớp không?**
   - Không. Val F1 giảm đơn điệu theo q (0,834 → 0,670), vì mô hình chưa quá khớp (gap 0,027, val loss còn giảm).
   - Nên dùng khi train loss tiếp tục giảm còn val loss tăng dần (ví dụ mạng lớn huấn luyện rất lâu), và chọn q theo val.
3. **Gradient clipping giải quyết vấn đề gì?**
   - Nó giải quyết các bước có gradient đột ngột rất lớn.
   - Bằng chứng: ở lr 3 không clip có gai grad_norm 7 814, sau đó mạng sụp về dự đoán hằng (F1 0,094). Cùng lr đó có clip c = 0,346 thì gai bị chặn (max 3,7) và mạng không sụp.
   - Ở lr thường không có gai lớn (max 2,84), nên clip chỉ làm bước ngắn lại và F1 giảm 0,011.
4. **Mixed precision có nhanh hơn không?**
   - Không: FP16 chậm hơn 42%, BF16 chậm hơn 20% trên T4, bộ nhớ đỉnh như nhau.
   - Lý do: với mạng và batch nhỏ như vậy, chi phí cố định mỗi bước (kernel launch, cast, GradScaler) lớn hơn phần tiết kiệm từ matmul FP16. BF16 lại không có phần cứng trên T4.
5. **Vì sao khởi tạo toàn số 0 hỏng? He khác Xavier thế nào?**
   - W = 0 làm mọi kích hoạt ẩn bằng 0 và mọi nơ-ron trong lớp giống hệt nhau. Gradient của W bằng 0 (hoặc giống nhau cho mọi nơ-ron), nên đối xứng không bao giờ bị phá. Chỉ bias lớp cuối học được, ứng với phân bố lớp: acc 0,4876, loss 1,205.
   - He dùng Var = 2/n_in, bù hệ số 1/2 mà ReLU làm mất. Xavier dùng 2/(n_in+n_out), hợp với kích hoạt tuyến tính/tanh.
   - Khác biệt nhỏ ở mạng 3 lớp (trong nhiễu), nhưng ở mạng 30 lớp ReLU, Xavier làm std co tới 4,6·10⁻⁶ trong khi He giữ được ~0,25.
6. **Loss không giảm sau 2 000 bước: 3 phép kiểm tra đầu tiên.**
   1. **Loss bước 0 so với ln 7.** Ví dụ: loss đứng ở 1,205 là "học phân bố lớp"; zeros cho loss bước 0 = ln 7 nhưng sau đó đứng yên. Nếu loss bước 0 lệch nhiều thì thang khởi tạo/chuẩn hoá có vấn đề.
   2. **Quá khớp 20 mẫu.** Phép kiểm tra rẻ này tách lỗi code (nhãn, softmax hai lần, quên `zero_grad`) khỏi lỗi tối ưu. Pipeline của tôi đưa loss về 6,6·10⁻⁶; nếu không làm được thì lỗi nằm ở code.
   3. **In grad_norm toàn cục và theo từng lớp.** Gradient bằng 0 ở W cho thấy khởi tạo đối xứng hoặc ReLU chết (`init-zeros`, `clip-none-highlr`). Gai rất lớn rồi phẳng cho thấy lr quá cao (lr 3: gai 7 814 rồi sụp; batch 2048 + lr 1,2 sụp). Nếu gradient đều nhưng nhỏ thì lr quá thấp (SGD lr 0,03 chậm).

   Sau đó mới xét kiến trúc/năng lực: nếu train và val cùng cao thì tăng độ rộng, như M-wide +0,014.

## 6. Hạn chế và điều bất ngờ

- **Khác dự đoán:**
  - Batch 128 cùng lr kém hơn, dù có nhiều bước hơn (do nhiễu gradient ở lr 0,3).
  - Tăng lr theo lô không warmup làm mô hình sụp.
  - MSE với lr ×3 còn tệ hơn.
  - Loss bước 0 của He cao hơn ln 7 (do thang của lớp ra).
  - Lr cao không clip không thành NaN mà "chết lặng", nên cờ `diverged` không bắt được. Cần thêm tiêu chí như "val loss ≈ entropy lớp".
- **Rủi ro trong thiết kế:**
  - Ngoài baseline và cấu hình cuối, mọi thí nghiệm chỉ có 1 seed. Các chênh lệch gần 2σ (Adam +0,008, BF16 −0,0074) chưa đủ để kết luận.
  - Lr tốt nhất của mọi bộ tối ưu nằm ở biên lưới, nên so sánh optimizer có thể đổi khi mở rộng lưới.
  - Các thí nghiệm batch giữ cùng số epoch nên khác số bước cập nhật.
  - σ đo từ 3 seed nên bản thân ước lượng σ cũng nhiễu.
  - Cấu hình cuối đổi 3 yếu tố cùng lúc nên không tách được đóng góp riêng của 40 epoch.
- **Nếu có thêm thời gian:**
  - Mở rộng lưới lr lên trên (SGD+momentum > 0,3, Adam > 3e-3).
  - Chạy 3 seed cho các thí nghiệm gần ngưỡng.
  - Thử warmup cho batch lớn, CE có trọng số lớp cho Aspen/Cottonwood, và scheduler cosine.

## 7. Phụ lục

- **File đã nộp:** `REPORT.md`, `experiments.xlsx` (39 dòng; sheet Seeds = base-s1..3; Summary có nhận xét), `predictions_eval.csv` (`final-s3`), `eval_result.json`, `figures/` (39 ảnh `<exp_id>.png` và 11 ảnh `compare_*.png`), `results/` (39 file `<exp_id>.json`), `code/` (`lab.ipynb`, `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`).
  - Ngoài ra có `predictions_eval_baseline.csv` và `eval_result_baseline.json`, chỉ để so baseline trên eval.
- **Thời gian chạy:** khoảng 25 phút trên T4 cho 39 lần chạy (~1,2 s/epoch với M-base batch 512; batch 128 ~4,8 s/epoch; cấu hình cuối 40 epoch ~1,4 s/epoch). Khi chạy lại, các lần đã có trên Drive được nạp từ cache chỉ trong vài phút.
