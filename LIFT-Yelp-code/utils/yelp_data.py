import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


CANONICAL_LABELS = ["food", "drink", "inside", "outside", "menu"]

CLASS_DEFINITIONS = {
    "food": "the main visible subject is prepared food, dishes, meals, desserts, snacks, or ingredients. If food is clearly the subject, choose food even when plates, bowls, tables, or drinks are also visible.",
    "drink": "the main visible subject is a beverage itself, such as coffee, tea, cocktails, wine, beer, juice, smoothies, or a prominently featured drink container. Do not choose drink merely because cups, glasses, bottles, or tableware appear in an indoor restaurant scene.",
    "inside": "the main visible subject is the restaurant interior, such as dining rooms, counters, tables, decor, indoor walls, staff areas, atmosphere, or indoor seating. If the image mainly shows the room or seating area, choose inside even if food or drink containers are present.",
    "outside": "the main visible subject is the restaurant exterior, storefront, street view, entrance, outdoor sign, patio, or outdoor seating. If the image mainly shows the building or outdoor area, choose outside even if people, tables, food, or drinks are present.",
    "menu": "the main visible subject is a menu, menu board, printed menu, price list, ordering screen, or ordering board. Choose menu only when menu text or a menu object is central and readable or clearly intended as the subject.",
}


@dataclass
class YelpRecord:
    photo_id: str
    relative_image_path: str
    image_path: str
    label: int
    label_name: str
    caption: str = ""


def read_classnames(classnames_path: os.PathLike) -> List[str]:
    classnames = []
    with open(classnames_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 2:
                continue
            classnames.append(" ".join(parts[1:]))
    return classnames


def read_caption_map(text_path: os.PathLike) -> Dict[str, str]:
    captions = {}
    if not text_path or not Path(text_path).exists():
        return captions

    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or parts[0] == "":
                continue
            captions[parts[0]] = parts[1] if len(parts) > 1 else ""
    return captions


def read_yelp_records(
    split: str,
    data_dir: os.PathLike = "datasets/Yelp",
    image_root: Optional[os.PathLike] = None,
    include_caption: bool = True,
) -> List[YelpRecord]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split {split!r}; expected train, val, or test.")

    data_dir = Path(data_dir)
    split_path = data_dir / f"Yelp_{split}.txt"
    text_path = data_dir / f"Yelp_{split}_text.txt"
    classnames = read_classnames(data_dir / "classnames.txt")
    caption_map = read_caption_map(text_path) if include_caption else {}
    image_root = Path(image_root) if image_root is not None else Path(".")

    records: List[YelpRecord] = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rel_path = parts[0]
            label = int(parts[1])
            photo_id = Path(rel_path).stem
            records.append(
                YelpRecord(
                    photo_id=photo_id,
                    relative_image_path=rel_path,
                    image_path=str(image_root / rel_path),
                    label=label,
                    label_name=classnames[label],
                    caption=caption_map.get(photo_id, ""),
                )
            )
    return records


def infer_yelp_image_root(config_path: os.PathLike = "configs/data/yelp_lt.yaml") -> Optional[str]:
    path = Path(config_path)
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("root:"):
                value = stripped.split(":", 1)[1].strip().strip("'\"")
                return value or None
    return None


def labels_to_names(labels: Iterable[int], classnames: List[str]) -> List[str]:
    return [classnames[int(label)] for label in labels]


def names_to_labels(classnames: List[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(classnames)}
