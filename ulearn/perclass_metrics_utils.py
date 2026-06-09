# /code/Fed/ulearn/perclass_metrics_utils.py

import csv
import torch
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def _sanitize_logits_tensor(x: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(x).all():
        print("[perclass_metrics_utils] Warning: non-finite logits detected, applying nan_to_num before metrics.")
    return torch.nan_to_num(x, nan=0.0, posinf=30.0, neginf=-30.0)


def compute_per_class_metrics(all_preds_tensor, all_targs_tensor, threshold=0.5):
    """
    根据 logits + targets，计算每个类别的 AP / P / R / F1

    参数：
        all_preds_tensor: (N, L) 的 logit 张量（模型输出，未过 sigmoid）
        all_targs_tensor: (N, L) 的 0/1 标签张量
        threshold:       把概率二值化的阈值（默认 0.5）

    返回：
        per_class: 长度为 L 的 list，每个元素是：
            {
              "AP": float,
              "P": float,
              "R": float,
              "F1": float,
            }
    """
    # logits -> 概率
    probs = torch.sigmoid(_sanitize_logits_tensor(all_preds_tensor)).detach().cpu().numpy()   # (N, L)
    targets = all_targs_tensor.detach().cpu().numpy()                # (N, L)

    num_labels = probs.shape[1]
    per_class = []

    for c in range(num_labels):
        y_true = targets[:, c]
        y_score = probs[:, c]

        # AP
        if y_true.sum() == 0:
            ap = 0.0
        else:
            ap = average_precision_score(y_true, y_score)

        # 概率 → 0/1 预测
        y_pred = (y_score >= threshold).astype(int)

        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        per_class.append({
            "AP":  float(ap),
            "P":   float(p),
            "R":   float(r),
            "F1":  float(f1),
        })

    return per_class


# ========= 新增：带样本子集 mask 的 Hamming / Subset =========
def compute_hamming_and_subset_accuracy(
    all_preds_tensor,
    all_targs_tensor,
    threshold: float = 0.5,
    sample_mask: torch.Tensor = None,
):
    """
    计算 Hamming accuracy 和 Subset accuracy（严格多标签 acc）

    参数：
        all_preds_tensor: (N, L) logits，未过 sigmoid
        all_targs_tensor: (N, L) 0/1 标签
        threshold:        float，把概率二值化的阈值（默认 0.5）
        sample_mask:      (N,) bool / byte 张量，表示选哪些样本参与统计；
                          若为 None，则在全部样本上统计。

    返回：
        dict:
          {
            "hamming_acc": float,
            "subset_acc":  float,
          }
    """
    # 先复制到同设备
    preds = _sanitize_logits_tensor(all_preds_tensor.detach())
    targs = all_targs_tensor.detach()

    # 按样本子集筛选
    if sample_mask is not None:
        sample_mask = sample_mask.to(preds.device)
        if sample_mask.dtype != torch.bool:
            sample_mask = sample_mask.bool()
        # 如果一个样本都没有，直接返回 0
        if sample_mask.sum().item() == 0:
            return {
                "hamming_acc": 0.0,
                "subset_acc":  0.0,
            }
        preds = preds[sample_mask]
        targs = targs[sample_mask]

    # logits -> prob -> 0/1 预测
    probs = torch.sigmoid(preds)                 # (N_sub, L)
    preds_bin = (probs >= threshold).long()      # (N_sub, L)
    targs_bin = targs.long()                     # (N_sub, L)

    # Hamming accuracy：每个 (样本, 类别) 当成一个二分类，整体正确比例
    correct_matrix = (preds_bin == targs_bin).float()     # (N_sub, L)
    hamming_acc = correct_matrix.mean().item()

    # Subset accuracy：整行标签完全一致才算对
    per_sample_correct = correct_matrix.all(dim=1)        # (N_sub,)
    subset_acc = per_sample_correct.float().mean().item()

    return {
        "hamming_acc": hamming_acc,
        "subset_acc":  subset_acc,
    }
# ============================= 新增部分结束 =============================


def _save_before_after_csv(per_before, per_after, csv_path, class_names=None, forget_cls=None):
    """
    内部函数：把前后 per-class 指标保存到 CSV 里。
    """
    num_labels = len(per_before)
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_labels)]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cls_idx", "cls_name",
            "AP_before", "AP_after", "ΔAP",
            "P_before",  "P_after",  "ΔP",
            "R_before",  "R_after",  "ΔR",
            "F1_before", "F1_after", "ΔF1",
            "is_forget_cls"
        ])

        for c in range(num_labels):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            b = per_before[c]
            a = per_after[c]

            writer.writerow([
                c, name,
                f"{b['AP']:.3f}",  f"{a['AP']:.3f}",  f"{(a['AP']  - b['AP']):+.3f}",
                f"{b['P']:.3f}",   f"{a['P']:.3f}",   f"{(a['P']   - b['P']):+.3f}",
                f"{b['R']:.3f}",   f"{a['R']:.3f}",   f"{(a['R']   - b['R']):+.3f}",
                f"{b['F1']:.3f}",  f"{a['F1']:.3f}",  f"{(a['F1']  - b['F1']):+.3f}",
                1 if (forget_cls is not None and c == forget_cls) else 0
            ])


def summarize_before_after(
    all_preds_before, all_targs_before,
    all_preds_after,  all_targs_after,
    class_names=None,
    forget_cls=None,
    csv_path=None,
    threshold=0.5,
):
    """
    主接口：给你 before/after 的 logits 和 targets，算出所有类别的
    AP/P/R/F1 变化，并可选保存成 CSV。
    """
    per_before = compute_per_class_metrics(all_preds_before, all_targs_before, threshold=threshold)
    per_after  = compute_per_class_metrics(all_preds_after,  all_targs_after,  threshold=threshold)

    if csv_path is not None:
        _save_before_after_csv(per_before, per_after, csv_path,
                               class_names=class_names, forget_cls=forget_cls)

    return per_before, per_after


def summarize_forget_vs_others(
    all_preds_before, all_targs_before,
    all_preds_after,  all_targs_after,
    forget_cls: int,
    class_names=None,
    csv_path: str = None,
    threshold: float = 0.5,
):
    """
    对比：
      1) 目标类 forget_cls 在忘却前后的 AP / P / R / F1 变化
      2) 所有非目标类的 AP / P / R / F1 平均变化
      3) 只含目标类样本上的 Hamming / Subset（before/after）
      4) 不含目标类样本上的 Hamming / Subset（before/after）

    返回：
        dict，包含：
          - 'forget_cls_before' / 'forget_cls_after'
          - 'others_mean_before' / 'others_mean_after'
          - 'target_subset_hamming_before' / 'target_subset_hamming_after'
          - 'others_subset_hamming_before' / 'others_subset_hamming_after'
    """

    # 1) 先算每个类别的 AP/P/R/F1（before/after）
    per_before = compute_per_class_metrics(all_preds_before, all_targs_before,
                                           threshold=threshold)
    per_after  = compute_per_class_metrics(all_preds_after,  all_targs_after,
                                           threshold=threshold)

    num_labels = len(per_before)
    assert 0 <= forget_cls < num_labels, "forget_cls 超出类别范围"

    # 2) 目标类指标
    b_f = per_before[forget_cls]
    a_f = per_after[forget_cls]

    # 3) 所有非目标类的平均指标（AP/P/R/F1）
    other_idx = [i for i in range(num_labels) if i != forget_cls]

    def _avg_metric(per_list, key, indices):
        vals = [per_list[i][key] for i in indices]
        return float(np.mean(vals)) if len(vals) > 0 else 0.0

    others_before = {
        "AP":  _avg_metric(per_before, "AP",  other_idx),
        "P":   _avg_metric(per_before, "P",   other_idx),
        "R":   _avg_metric(per_before, "R",   other_idx),
        "F1":  _avg_metric(per_before, "F1",  other_idx),
    }
    others_after = {
        "AP":  _avg_metric(per_after,  "AP",  other_idx),
        "P":   _avg_metric(per_after,  "P",   other_idx),
        "R":   _avg_metric(per_after,  "R",   other_idx),
        "F1":  _avg_metric(per_after,  "F1",  other_idx),
    }

    # 4) 构造“含目标类 / 不含目标类”的样本子集 mask
    #    注意：标签 before / after 是同一份，所以只用 before 的 targets 来造 mask 即可
    targs_before = all_targs_before.detach()
    mask_with_target    = (targs_before[:, forget_cls] == 1)   # 只含目标类样本
    mask_without_target = (targs_before[:, forget_cls] == 0)   # 不含目标类样本

    # 5) 在“含目标类的样本子集”上，算 Hamming / Subset（before / after）
    target_subset_before = compute_hamming_and_subset_accuracy(
        all_preds_before, all_targs_before,
        threshold=threshold,
        sample_mask=mask_with_target,
    )
    target_subset_after = compute_hamming_and_subset_accuracy(
        all_preds_after, all_targs_after,
        threshold=threshold,
        sample_mask=mask_with_target,
    )

    # 6) 在“不含目标类的样本子集”上，算 Hamming / Subset（before / after）
    others_subset_before = compute_hamming_and_subset_accuracy(
        all_preds_before, all_targs_before,
        threshold=threshold,
        sample_mask=mask_without_target,
    )
    others_subset_after = compute_hamming_and_subset_accuracy(
        all_preds_after, all_targs_after,
        threshold=threshold,
        sample_mask=mask_without_target,
    )

    # 7) 可选写 CSV：两行——目标类 + 非目标类平均
    if csv_path is not None:
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_labels)]
        forget_name = class_names[forget_cls] if forget_cls < len(class_names) \
                     else f"class_{forget_cls}"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tag", "cls_idx", "cls_name",
                "AP_before", "AP_after", "ΔAP",
                "P_before",  "P_after",  "ΔP",
                "R_before",  "R_after",  "ΔR",
                "F1_before", "F1_after", "ΔF1",
                "Hamming_before", "Hamming_after", "ΔHamming",
                "Subset_before",  "Subset_after",  "ΔSubset",
            ])

            # 行 1：目标类（Hamming/Subset 只在“含目标类样本”子集上统计）
            th_b = target_subset_before["hamming_acc"]
            th_a = target_subset_after["hamming_acc"]
            ts_b = target_subset_before["subset_acc"]
            ts_a = target_subset_after["subset_acc"]

            writer.writerow([
                "forget_cls", forget_cls, forget_name,
                f"{b_f['AP']:.3f}",  f"{a_f['AP']:.3f}",  f"{(a_f['AP']  - b_f['AP']):+.3f}",
                f"{b_f['P']:.3f}",   f"{a_f['P']:.3f}",   f"{(a_f['P']   - b_f['P']):+.3f}",
                f"{b_f['R']:.3f}",   f"{a_f['R']:.3f}",   f"{(a_f['R']   - b_f['R']):+.3f}",
                f"{b_f['F1']:.3f}",  f"{a_f['F1']:.3f}",  f"{(a_f['F1']  - b_f['F1']):+.3f}",
                f"{th_b:.3f}",       f"{th_a:.3f}",       f"{(th_a - th_b):+.3f}",
                f"{ts_b:.3f}",       f"{ts_a:.3f}",       f"{(ts_a - ts_b):+.3f}",
            ])

            # 行 2：非目标类平均（Hamming/Subset 只在“不含目标类样本”子集上统计）
            oh_b = others_subset_before["hamming_acc"]
            oh_a = others_subset_after["hamming_acc"]
            os_b = others_subset_before["subset_acc"]
            os_a = others_subset_after["subset_acc"]

            writer.writerow([
                "others_mean", "-", "others",
                f"{others_before['AP']:.3f}",  f"{others_after['AP']:.3f}",
                f"{(others_after['AP']  - others_before['AP']):+.3f}",
                f"{others_before['P']:.3f}",   f"{others_after['P']:.3f}",
                f"{(others_after['P']   - others_before['P']):+.3f}",
                f"{others_before['R']:.3f}",   f"{others_after['R']:.3f}",
                f"{(others_after['R']   - others_before['R']):+.3f}",
                f"{others_before['F1']:.3f}",  f"{others_after['F1']:.3f}",
                f"{(others_after['F1']  - others_before['F1']):+.3f}",
                f"{oh_b:.3f}",                 f"{oh_a:.3f}",       f"{(oh_a - oh_b):+.3f}",
                f"{os_b:.3f}",                 f"{os_a:.3f}",       f"{(os_a - os_b):+.3f}",
            ])

    return {
        "forget_cls_before": b_f,
        "forget_cls_after": a_f,
        "others_mean_before": others_before,
        "others_mean_after":  others_after,
        "target_subset_hamming_before": target_subset_before,
        "target_subset_hamming_after":  target_subset_after,
        "others_subset_hamming_before": others_subset_before,
        "others_subset_hamming_after":  others_subset_after,
    }
