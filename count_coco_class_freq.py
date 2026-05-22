import os
import pickle
import argparse
import numpy as np
import csv


# 如果你自己的类别顺序和下面不一样，就把这个列表换成你项目里的 COCO 类别顺序
COCO80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def count_coco_classes(train_data_file, num_labels=80, save_csv=None):
    print(f"Loading split data from: {train_data_file}")
    with open(train_data_file, "rb") as f:
        split_data = pickle.load(f)

    print(f"Total images in train split: {len(split_data)}")

    counts = np.zeros(num_labels, dtype=np.int64)

    for item in split_data:
        labels = np.array(item["objects"], dtype=np.int64)  # 0/1 多标签向量
        if labels.shape[0] != num_labels:
            raise ValueError(
                f"num_labels mismatch: got {labels.shape[0]}, expected {num_labels}"
            )
        counts += labels

    print("\n=== COCO train per-class counts ===")
    print(f"{'idx':>3}  {'class_name':<15}  {'count':>8}")
    print("-" * 32)
    for i in range(num_labels):
        cls_name = COCO80_CLASSES[i] if i < len(COCO80_CLASSES) else f"cls_{i}"
        print(f"{i:3d}  {cls_name:<15}  {counts[i]:8d}")

    if save_csv is not None:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cls_idx", "cls_name", "count"])
            for i in range(num_labels):
                cls_name = COCO80_CLASSES[i] if i < len(COCO80_CLASSES) else f"cls_{i}"
                writer.writerow([i, cls_name, int(counts[i])])
        print(f"\nPer-class counts saved to: {save_csv}")

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Count per-class sample numbers in COCO80 train split"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="你的 data_root 路径（里面有 coco 文件夹）",
    )
    parser.add_argument(
        "--train_data_name",
        type=str,
        default="train.data",
        help="train.data 文件名（默认和你代码一致）",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=80,
        help="类别数（COCO80 默认 80）",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="可选：把结果保存到 csv，例如 results/coco_class_counts.csv",
    )

    args = parser.parse_args()

    coco_root = os.path.join(args.data_root, "coco")
    train_data_file = os.path.join(coco_root, args.train_data_name)

    if not os.path.exists(train_data_file):
        raise FileNotFoundError(f"train.data not found: {train_data_file}")

    count_coco_classes(
        train_data_file=train_data_file,
        num_labels=args.num_labels,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()