import os

from torchvision import transforms
from dataloaders.coco80_dataset import Coco80Dataset   # 路径按你自己的

transform = transforms.Compose([
    transforms.Resize((448, 448)),   # 或者你训练时用的尺寸
    transforms.ToTensor(),
])

coco_root = "/code/Fed/data/coco"

train_ds = Coco80Dataset(
    split="train",
    num_labels=80,
    data_file=os.path.join(coco_root, "train.data"),
    img_root=os.path.join(coco_root, "train2014"),
    annotation_dir=None,
    max_samples=10,
    transform=transform,     # ⭐ 有 transform，就会返回 tensor
    known_labels=0,
    testing=False
)

sample = train_ds[0]
print(type(sample['image']))
print(sample['image'].shape, sample['labels'].shape)