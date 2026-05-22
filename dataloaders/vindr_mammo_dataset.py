import ast
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataloaders.data_utils import get_unk_mask_indices


# VinDr-Mammo 的多标签方案分成两档：
# 1. breast:
#    只使用乳腺级标签，包含 BI-RADS 1~5 + Density A~D，共 9 类。
# 2. breast_findings:
#    在 breast 的基础上，再拼上 finding_annotations 里的病灶类别，共 20 类。
#
# 第二种更像真正的“多标签”医学图像分类：
# 一张图既有乳腺级属性，也可能同时带一个或多个病灶标签。
VINDR_BREAST_CLASSES: List[str] = [
    "BI-RADS 1",
    "BI-RADS 2",
    "BI-RADS 3",
    "BI-RADS 4",
    "BI-RADS 5",
    "Density A",
    "Density B",
    "Density C",
    "Density D",
]

VINDR_FINDING_CLASSES: List[str] = [
    "Architectural Distortion",
    "Asymmetry",
    "Focal Asymmetry",
    "Global Asymmetry",
    "Mass",
    "Nipple Retraction",
    "No Finding",
    "Skin Retraction",
    "Skin Thickening",
    "Suspicious Calcification",
    "Suspicious Lymph Node",
]

VINDR_LABEL_MODES = {
    "breast": VINDR_BREAST_CLASSES,
    "breast_findings": VINDR_BREAST_CLASSES + VINDR_FINDING_CLASSES,
}


def get_vindr_class_names(label_mode: str = "breast_findings") -> List[str]:
    if label_mode not in VINDR_LABEL_MODES:
        raise ValueError(f"Unsupported VinDr label_mode: {label_mode}")
    return list(VINDR_LABEL_MODES[label_mode])


def _find_existing_file(root: str, candidates: List[str]) -> str:
    """兼容正常文件名和本地 Finder 自动加上的 (1) 后缀。"""
    for name in candidates:
        path = os.path.join(root, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of the files exist under {root}: {candidates}")


def _normalize_birads(value) -> str:
    text = str(value).strip().upper()
    for digit in ["1", "2", "3", "4", "5"]:
        if digit in text:
            return digit
    raise ValueError(f"Unsupported breast_birads value: {value}")


def _normalize_density(value) -> str:
    text = str(value).strip().upper()
    for letter in ["A", "B", "C", "D"]:
        if text.endswith(letter) or f" {letter}" in text or text == letter:
            return letter
    raise ValueError(f"Unsupported breast_density value: {value}")


def _parse_finding_categories(value) -> List[str]:
    """把 finding_categories 列里形如 \"['Mass']\" 的字符串解成列表。"""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value]
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = text
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed]
    return [str(parsed).strip()]


def _load_dicom_as_pil(path: str) -> Image.Image:
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    except ImportError as exc:
        raise ImportError(
            "VinDr-Mammo 原始文件是 DICOM。请先安装 `pydicom`，例如：pip install pydicom"
        ) from exc

    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array
    try:
        pixels = apply_voi_lut(pixels, ds)
    except Exception:
        pass

    pixels = pixels.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels

    pixels -= pixels.min()
    denom = max(float(pixels.max()), 1e-6)
    pixels = (pixels / denom * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(pixels).convert("RGB")


def _load_image(path: str, transform):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".dicom":
        image = _load_dicom_as_pil(path)
    else:
        image = Image.open(path).convert("RGB")

    if transform is not None:
        image = transform(image)
    return image


class VinDrMammoDataset(Dataset):
    """VinDr-Mammo 图像级多标签分类数据集。

    当前 loader 是独立的，不影响 VOC/COCO。
    它直接从官方 CSV 构造 multi-hot 标签，适配现有 CTran 多标签训练流。
    """

    def __init__(
        self,
        vindr_root,
        split,
        transform=None,
        known_labels=0,
        testing=False,
        max_samples=-1,
        label_mode="breast_findings",
    ):
        self.vindr_root = vindr_root
        self.split = split
        self.transform = transform
        self.known_labels = known_labels
        self.testing = testing
        self.label_mode = label_mode
        self.class_names = get_vindr_class_names(label_mode)
        self.num_labels = len(self.class_names)
        self.epoch = 1

        breast_ann_path = _find_existing_file(
            vindr_root,
            ["breast-level_annotations.csv", "breast-level_annotations(1).csv"],
        )
        finding_ann_path = _find_existing_file(
            vindr_root,
            ["finding_annotations.csv", "finding_annotations(1).csv"],
        )

        breast_df = pd.read_csv(breast_ann_path)
        if "split" not in breast_df.columns:
            raise ValueError("VinDr-Mammo breast-level_annotations.csv 缺少 `split` 列。")

        breast_df = breast_df[breast_df["split"].astype(str).str.lower() == split.lower()].copy()
        if max_samples != -1:
            breast_df = breast_df.iloc[:max_samples].copy()
        breast_df.reset_index(drop=True, inplace=True)

        required_cols = [
            "study_id",
            "series_id",
            "image_id",
            "laterality",
            "view_position",
            "breast_birads",
            "breast_density",
        ]
        for col in required_cols:
            if col not in breast_df.columns:
                raise ValueError(f"VinDr-Mammo breast-level_annotations.csv 缺少 `{col}` 列。")

        finding_df = pd.read_csv(finding_ann_path)
        finding_df = finding_df[finding_df["split"].astype(str).str.lower() == split.lower()].copy()
        grouped_findings: Dict[str, List[str]] = {}
        if "image_id" in finding_df.columns and "finding_categories" in finding_df.columns:
            for _, row in finding_df.iterrows():
                image_id = str(row["image_id"]).strip()
                grouped_findings.setdefault(image_id, [])
                grouped_findings[image_id].extend(_parse_finding_categories(row["finding_categories"]))

        finding_to_idx = {name: idx for idx, name in enumerate(VINDR_FINDING_CLASSES)}

        self.split_data = []
        self.labels = []
        for _, row in breast_df.iterrows():
            image_id = str(row["image_id"]).strip()
            study_id = str(row["study_id"]).strip()
            series_id = str(row["series_id"]).strip()

            label_vec = np.zeros(self.num_labels, dtype=np.float32)

            # 乳腺级标签部分：
            # BI-RADS 1~5 占前 5 维，Density A~D 紧接着 4 维。
            birads = _normalize_birads(row["breast_birads"])
            density = _normalize_density(row["breast_density"])
            label_vec[int(birads) - 1] = 1.0
            label_vec[5 + (ord(density) - ord("A"))] = 1.0

            # 病灶级标签部分：
            # 仅当 label_mode=breast_findings 时追加到后半段。
            if self.label_mode == "breast_findings":
                findings = grouped_findings.get(image_id, [])
                for finding_name in findings:
                    if finding_name in finding_to_idx:
                        label_vec[len(VINDR_BREAST_CLASSES) + finding_to_idx[finding_name]] = 1.0

            image_candidates = [
                os.path.join(vindr_root, "images", study_id, f"{image_id}.dicom"),
                os.path.join(vindr_root, "images", study_id, f"{image_id}.png"),
                os.path.join(vindr_root, "images", study_id, f"{image_id}.jpg"),
                os.path.join(vindr_root, "images", study_id, series_id, f"{image_id}.dicom"),
                os.path.join(vindr_root, "images", study_id, series_id, f"{image_id}.png"),
                os.path.join(vindr_root, "images", study_id, series_id, f"{image_id}.jpg"),
            ]
            image_path = None
            for candidate in image_candidates:
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            if image_path is None:
                raise FileNotFoundError(
                    f"找不到 VinDr-Mammo 图像文件，尝试过这些路径：{image_candidates}"
                )

            self.split_data.append(
                {
                    "image_id": image_id,
                    "study_id": study_id,
                    "series_id": series_id,
                    "laterality": str(row["laterality"]),
                    "view_position": str(row["view_position"]),
                    "image_path": image_path,
                    "objects": label_vec,
                }
            )
            self.labels.append(label_vec)

        self.labels = np.asarray(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, idx):
        sample_info = self.split_data[idx]
        image = _load_image(sample_info["image_path"], self.transform)
        labels = torch.tensor(sample_info["objects"], dtype=torch.float32)

        unk_mask_indices = get_unk_mask_indices(
            image,
            self.testing,
            self.num_labels,
            self.known_labels,
            self.epoch,
        )
        mask = labels.clone()
        mask.scatter_(0, torch.tensor(unk_mask_indices).long(), -1)

        return {
            "image": image,
            "labels": labels,
            "mask": mask,
            "imageIDs": sample_info["image_id"],
        }
