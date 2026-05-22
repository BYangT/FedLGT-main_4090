# per_class_report.py
import csv
import numpy as np
import torch

try:
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[per_class_report] sklearn not found, AP 将用简单近似（非严格 mAP）")

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.asarray(x)

def _per_class_stats(y_true, y_logits, thr=0.5, eps=1e-8):
    """
    y_true:  (N, L) 0/1
    y_logits: (N, L) 未过 sigmoid
    返回：
      ap:  (L,)
      p:   (L,)
      r:   (L,)
      f1:  (L,)
    """
    y_true = _to_numpy(y_true).astype(np.float32)
    y_logits = _to_numpy(y_logits).astype(np.float32)

    N, L = y_true.shape
    # 概率
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))
    y_pred = (y_prob >= thr).astype(np.float32)

    ap = np.zeros(L, dtype=np.float32)
    p  = np.zeros(L, dtype=np.float32)
    r  = np.zeros(L, dtype=np.float32)
    f1 = np.zeros(L, dtype=np.float32)

    for j in range(L):
        yj_true = y_true[:, j]
        yj_prob = y_prob[:, j]
        yj_pred = y_pred[:, j]

        # 1) AP
        if SKLEARN_AVAILABLE and (yj_true.sum() > 0):
            try:
                ap[j] = average_precision_score(yj_true, yj_prob)
            except Exception:
                ap[j] = 0.0
        else:
            # 没有 sklearn 时的简易替代（不严格）
            ap[j] = (yj_true * yj_prob).sum() / (yj_true.sum() + eps)

        # 2) P / R / F1
        tp = ((yj_pred == 1) & (yj_true == 1)).sum()
        fp = ((yj_pred == 1) & (yj_true == 0)).sum()
        fn = ((yj_pred == 0) & (yj_true == 1)).sum()

        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1_j = 2 * prec * rec / (prec + rec + eps)

        p[j]  = prec
        r[j]  = rec
        f1[j] = f1_j

    return ap, p, r, f1


def save_voc_per_class_report_csv(
    all_targs_before,
    all_preds_before,
    all_targs_after,
    all_preds_after,
    class_names,
    csv_path,
    thr: float = 0.5,
):
    """
    生成 VOC 风格的 per-class 报表：
      - 每一类：AP / P / R / F1（before/after + delta）
      - 每一类：Hamming acc / Subset acc（before/after + delta）
    其中 Hamming / Subset 是在「只包含该类的样本子集」上计算的。

    参数:
        all_targs_before: (N, L) tensor，遗忘前/恢复前的标签
        all_preds_before: (N, L) tensor，遗忘前/恢复前的 logits
        all_targs_after:  (N, L) tensor，遗忘后/恢复后的标签（一般与 before 相同）
        all_preds_after:  (N, L) tensor，遗忘后/恢复后的 logits
        class_names:      list[str]，类别名
        csv_path:         str，输出 csv 路径
        thr:              float，二值化阈值
    """

    # ---- 转成 numpy ----
    logits_b = all_preds_before.detach().cpu()
    logits_a = all_preds_after.detach().cpu()
    targs_b = all_targs_before.detach().cpu()
    targs_a = all_targs_after.detach().cpu()

    assert logits_b.shape == logits_a.shape == targs_b.shape == targs_a.shape
    N, L = targs_b.shape

    probs_b = torch.sigmoid(logits_b).numpy()          # (N, L)
    probs_a = torch.sigmoid(logits_a).numpy()          # (N, L)
    y_true_b = targs_b.numpy().astype(np.int32)        # (N, L)
    y_true_a = targs_a.numpy().astype(np.int32)        # 通常和 y_true_b 一样

    # 全标签二值化，用于 Hamming / Subset 计算
    y_pred_b_all = (probs_b >= thr).astype(np.int32)   # (N, L)
    y_pred_a_all = (probs_a >= thr).astype(np.int32)   # (N, L)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cls_idx", "cls_name",
            "AP_before", "AP_after", "AP_delta",
            "P_before",  "P_after",  "P_delta",
            "R_before",  "R_after",  "R_delta",
            "F1_before", "F1_after", "F1_delta",
            "Hamming_before", "Hamming_after", "Hamming_delta",
            "Subset_before",  "Subset_after",  "Subset_delta",
        ])

        for c in range(L):
            name = class_names[c] if c < len(class_names) else f"class_{c}"

            # ---------- 1) per-class AP / P / R / F1 ----------
            y_true_c = y_true_b[:, c]          # (N,)
            y_score_b = probs_b[:, c]
            y_score_a = probs_a[:, c]

            # AP
            if y_true_c.sum() == 0:
                ap_b = 0.0
                ap_a = 0.0
            else:
                ap_b = average_precision_score(y_true_c, y_score_b)
                ap_a = average_precision_score(y_true_c, y_score_a)

            # 概率 -> 二值预测（只看这一类）
            y_pred_b = (y_score_b >= thr).astype(np.int32)
            y_pred_a = (y_score_a >= thr).astype(np.int32)

            p_b, r_b, f1_b, _ = precision_recall_fscore_support(
                y_true_c, y_pred_b, average="binary", zero_division=0
            )
            p_a, r_a, f1_a, _ = precision_recall_fscore_support(
                y_true_c, y_pred_a, average="binary", zero_division=0
            )

            # ---------- 2) per-class Hamming / Subset ----------
            # 只在“该类为正”的样本子集上计算
            mask_pos = (y_true_c == 1)
            if mask_pos.sum() == 0:
                h_b = h_a = s_b = s_a = 0.0
            else:
                # 取出这一类为正的样本，对应的全标签预测 / 真实标签
                pred_b_sub = y_pred_b_all[mask_pos]    # (N_pos, L)
                pred_a_sub = y_pred_a_all[mask_pos]    # (N_pos, L)
                true_sub   = y_true_b[mask_pos]        # (N_pos, L)

                correct_b = (pred_b_sub == true_sub).astype(np.float32)
                correct_a = (pred_a_sub == true_sub).astype(np.float32)

                # Hamming：所有 (样本, 类别) bit 的正确率
                h_b = float(correct_b.mean())
                h_a = float(correct_a.mean())

                # Subset：这一子集里，整行完全一致的比例
                s_b = float(correct_b.all(axis=1).mean())
                s_a = float(correct_a.all(axis=1).mean())

            writer.writerow([
                c, name,
                f"{ap_b:.3f}",  f"{ap_a:.3f}",  f"{(ap_a  - ap_b):+.3f}",
                f"{p_b:.3f}",   f"{p_a:.3f}",   f"{(p_a   - p_b):+.3f}",
                f"{r_b:.3f}",   f"{r_a:.3f}",   f"{(r_a   - r_b):+.3f}",
                f"{f1_b:.3f}",  f"{f1_a:.3f}",  f"{(f1_a  - f1_b):+.3f}",
                f"{h_b:.3f}",   f"{h_a:.3f}",   f"{(h_a   - h_b):+.3f}",
                f"{s_b:.3f}",   f"{s_a:.3f}",   f"{(s_a   - s_b):+.3f}",
            ])