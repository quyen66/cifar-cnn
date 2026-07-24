# L1/L2 Deep Analysis — Report

## 1. L1 có thừa không?
- **A2 (đóng góp trực tiếp, L1 bắt được mà L2 bỏ sót): 0 client-round** (kỳ vọng cấu trúc = 0, xem code `_apply_decision_matrix`: L1 REJECTED → final REJECTED luôn).
- **A3 (đóng góp gián tiếp qua FLAG): 26/40 ô (dataset,alpha,cosine_bin) có |flag_effect| > 0.2** — cùng dải cosine, tỉ lệ bị loại cuối cùng KHÁC hẳn giữa client bị L1 FLAG vs không FLAG.
    cifar-10 α=1.0 cosine∈0.80-0.85: reject_rate(FLAGGED)=1.00 vs reject_rate(L1 ACCEPTED)=0.00 (chênh +1.00)
    fashion-mnist α=0.5 cosine∈0.85-0.90: reject_rate(FLAGGED)=1.00 vs reject_rate(L1 ACCEPTED)=0.00 (chênh +1.00)
    fashion-mnist α=1.0 cosine∈0.85-0.90: reject_rate(FLAGGED)=1.00 vs reject_rate(L1 ACCEPTED)=0.00 (chênh +1.00)
    fashion-mnist α=0.5 cosine∈0.80-0.85: reject_rate(FLAGGED)=1.00 vs reject_rate(L1 ACCEPTED)=0.00 (chênh +1.00)
    cifar-10 α=0.1 cosine∈0.85-0.90: reject_rate(FLAGGED)=1.00 vs reject_rate(L1 ACCEPTED)=0.00 (chênh +1.00)
    cifar-10 α=0.1 cosine∈0.80-0.85: reject_rate(FLAGGED)=1.00 vs reject_rate(L1 ACCEPTED)=0.00 (chênh +1.00)
    cifar-10 α=1.0 cosine∈0.85-0.90: reject_rate(FLAGGED)=1.00 vs reject_rate(L1 ACCEPTED)=0.00 (chênh +1.00)
    fashion-mnist α=0.25 cosine∈0.80-0.85: reject_rate(FLAGGED)=1.00 vs reject_rate(L1 ACCEPTED)=0.01 (chênh +0.99)
- **KẾT LUẬN:** L1 THỪA về mặt đóng góp trực tiếp (không bắt riêng được ai — cấu trúc code đã đảm bảo L2 luôn bao trùm L1 REJECTED), NHƯNG KHÔNG THỪA về mặt cơ chế FLAG — L1 FLAG thay đổi ngưỡng chấp nhận (0.9 rescue thay vì chỉ cần 0.8 cơ bản), ảnh hưởng rõ tới quyết định cuối, đặc biệt vùng cosine 0.80-0.90.

## 2. Có suy giảm theo thời gian không?
- DR trung bình vòng 11-20 vs 41-50: 0.582 → 0.597 (ổn định)
- FPR trung bình vòng 11-20 vs 41-50: 0.078 → 0.079 (ổn định)
- Chi tiết theo từng (dataset,attack,alpha,round): xem B_dr_fpr_perround.csv; ngưỡng L1/L2 theo vòng: B_threshold_drift.csv; cosine benign trôi: B_benign_cosine_drift.csv.

## 3. Rescue sai bao nhiêu, sửa được không?
- Toàn ma trận: 1893 sự kiện rescue, **384 SAI (20.3%)**.
- C_rescue_separability.csv: cosine của ca đúng/sai gần như KHÔNG tách được (đã biết từ trước); distance tách rõ hơn nhiều → xem C_rescue_altrule.csv để biết siết distance_multiplier xuống bao nhiêu thì loại được rescue sai mà không mất rescue đúng.
- C_rescue_alpha_anomaly.csv: giải thích vì sao từng alpha khác nhau qua phân bố distance của attacker bị FLAG.

## 4. Mỗi ô sụp accuracy thuộc H1 (sót) hay H2 (đói)?
  - cifar-10 minmax α=0.1: **H1_SOT**
  - cifar-10 minmax α=0.25: **KHONG_RO**
  - cifar-10 minmax α=1.0: **KHONG_RO**
  - fashion-mnist alie α=0.1: **H1_SOT**
  - fashion-mnist alie α=0.25: **H1_SOT**
  - fashion-mnist alie α=0.5: **H1_SOT**
  - fashion-mnist minmax α=0.1: **H1_SOT**
  - fashion-mnist minmax α=0.25: **H1_SOT**
  - fashion-mnist minmax α=0.5: **H1_SOT**
  (KHONG_RO = model ĐÃ THẤP NGAY TỪ VÒNG 11 (round post-warmup đầu tiên), không có "vòng gãy" rõ để so trước/sau — bản thân việc thấp ngay từ đầu cũng là dấu hiệu bất thường, xem accuracy thô theo vòng trong round_meta.csv cho các ô này)
- 9 ô được xét là nghi sụp (acc cuối < 0.35 hoặc std giữa seed > 0.15).
- Chi tiết per-seed: D_seed_divergence.csv; per-round quanh điểm sụp: D_collapse_context.csv (⚠️ n_attacker_leaked là PROXY từ L1/L2, không phải danh sách chính xác client vào aggregation — xem giới hạn trong config_detected.txt).

## 5. Bảng DR/FPR đầy đủ
- cifar-10 / L1: DR trung bình toàn bộ ô = 0.297, FPR trung bình toàn bộ ô = 0.019
- fashion-mnist / L1: DR trung bình toàn bộ ô = 0.276, FPR trung bình toàn bộ ô = 0.029
- cifar-10 / L2: DR trung bình toàn bộ ô = 0.604, FPR trung bình toàn bộ ô = 0.114
- fashion-mnist / L2: DR trung bình toàn bộ ô = 0.570, FPR trung bình toàn bộ ô = 0.047
- cifar-10 / Final: DR trung bình toàn bộ ô = 0.604, FPR trung bình toàn bộ ô = 0.114
- fashion-mnist / Final: DR trung bình toàn bộ ô = 0.570, FPR trung bình toàn bộ ô = 0.047
- Chênh lệch FPR giữa 2 dataset LỚN NHẤT (cùng attack/alpha/module):
    slow_poison α=0.1 Final: ΔFPR=0.324 (cifar=0.435, fashion=0.111)
    slow_poison α=0.1 L2: ΔFPR=0.324 (cifar=0.435, fashion=0.111)
    alie α=0.1 Final: ΔFPR=0.281 (cifar=0.380, fashion=0.100)
    alie α=0.1 L2: ΔFPR=0.281 (cifar=0.380, fashion=0.100)
    gaussian_noise α=0.1 L2: ΔFPR=0.262 (cifar=0.286, fashion=0.024)
- Bảng đầy đủ: E_dr_fpr_full.csv, E_by_alpha.csv, E_dataset_gap.csv

## Cảnh báo tính toàn vẹn
- 12 ô thiếu log hoàn toàn: [('cifar-10', 'none', '0.1', '123'), ('cifar-10', 'none', '0.1', '42'), ('cifar-10', 'none', '0.1', '777'), ('cifar-10', 'none', '0.25', '123'), ('cifar-10', 'none', '0.25', '42'), ('cifar-10', 'none', '0.25', '777'), ('fashion-mnist', 'none', '0.1', '123'), ('fashion-mnist', 'none', '0.1', '42'), ('fashion-mnist', 'none', '0.1', '777'), ('fashion-mnist', 'none', '0.25', '123'), ('fashion-mnist', 'none', '0.25', '42'), ('fashion-mnist', 'none', '0.25', '777')]
