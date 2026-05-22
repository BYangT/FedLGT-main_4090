import os
import json
import pickle
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np

def build_split(ann_file, img_root, cat_id2idx):
    """
    根据 COCO 的 instances_*.json 构造一个 list[dict]:
      [{'file_name': xxx.jpg, 'objects': [0/1,...]}, ...]
    """
    print(f"Loading annotations from {ann_file}")
    with open(ann_file, "r") as f:
        coco = json.load(f)

    # 1) 建立 image_id -> file_name 映射
    img_id2name = {}
    for img in coco["images"]:
        img_id2name[img["id"]] = img["file_name"]

    # 2) 按 image_id 收集所有类别
    img_id2cats = defaultdict(set)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue  # 可以跳过 crowd 标注
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if cat_id not in cat_id2idx:
            continue
        idx = cat_id2idx[cat_id]
        img_id2cats[img_id].add(idx)

    # 3) 做成 list[dict]
    num_labels = len(cat_id2idx)
    split_data = []
    for img_id, fname in tqdm(img_id2name.items(), desc="build split"):
        labels = np.zeros(num_labels, dtype="float32")
        for idx in img_id2cats.get(img_id, []):
            labels[idx] = 1.0

        sample = {
            "file_name": fname,
            "objects": labels.tolist(),
        }
        split_data.append(sample)

    print(f"Total images in {os.path.basename(ann_file)}: {len(split_data)}")
    return split_data

def main(args):
    coco_root = args.coco_root

    ann_train = os.path.join(coco_root, "annotations", "instances_train2014.json")
    ann_val   = os.path.join(coco_root, "annotations", "instances_val2014.json")

    assert os.path.exists(ann_train), ann_train
    assert os.path.exists(ann_val), ann_val

    # 先读一个 json 拿到所有类别 id，做 cat_id -> [0..79] 的映射
    with open(ann_train, "r") as f:
        coco_train = json.load(f)

    cat_ids = sorted([c["id"] for c in coco_train["categories"]])
    cat_id2idx = {cid: i for i, cid in enumerate(cat_ids)}
    print("Num categories:", len(cat_id2idx))

    # ------- 训练集：train2014 -------
    train_split = build_split(ann_train,
                              img_root=os.path.join(coco_root, "train2014"),
                              cat_id2idx=cat_id2idx)
    train_data_path = os.path.join(coco_root, "train.data")
    with open(train_data_path, "wb") as f:
        pickle.dump(train_split, f)
    print("Saved train.data to", train_data_path)

    # ------- 验证+测试：这里先用 val2014 全部当作 val_test -------
    val_split = build_split(ann_val,
                            img_root=os.path.join(coco_root, "val2014"),
                            cat_id2idx=cat_id2idx)
    val_data_path = os.path.join(coco_root, "val_test.data")
    with open(val_data_path, "wb") as f:
        pickle.dump(val_split, f)
    print("Saved val_test.data to", val_data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco_root",
        type=str,
        default="/code/Fed/data/coco",   # 按你的路径改
    )
    args = parser.parse_args()
    main(args)