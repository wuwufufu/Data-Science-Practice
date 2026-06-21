import os
from collections import Counter

from .lt_data import LT_Dataset


class Yelp_LT(LT_Dataset):
    classnames_txt = "./datasets/Yelp/classnames.txt"
    train_txt = "./datasets/Yelp/Yelp_train.txt"
    val_txt = "./datasets/Yelp/Yelp_val.txt"
    test_txt = "./datasets/Yelp/Yelp_test.txt"

    def __init__(self, root, train=True, val=False, transform=None):
        if val:
            self.test_txt = self.val_txt
        super().__init__(root, train, transform)

        self.classnames = self.read_classnames()
        self.names = [self.classnames[label] for label in self.labels]

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        name = self.names[index]
        return image, label, name

    @classmethod
    def read_classnames(self):
        classnames = []
        with open(self.classnames_txt, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                if not parts:
                    continue
                classname = " ".join(parts[1:])
                classnames.append(classname)
        return classnames


class Yelp_MM_LT(LT_Dataset):
    classnames_txt = "./datasets/Yelp/classnames.txt"
    train_txt = "./datasets/Yelp/Yelp_train.txt"
    val_txt = "./datasets/Yelp/Yelp_val.txt"
    test_txt = "./datasets/Yelp/Yelp_test.txt"
    train_text_txt = "./datasets/Yelp/Yelp_train_text.txt"
    val_text_txt = "./datasets/Yelp/Yelp_val_text.txt"
    test_text_txt = "./datasets/Yelp/Yelp_test_text.txt"

    def __init__(self, root, train=True, val=False, transform=None):
        split_name = "train" if train else ("val" if val else "test")
        self.split_name = split_name
        if val:
            self.test_txt = self.val_txt
            self.test_text_txt = self.val_text_txt
        super().__init__(root, train, transform)

        if train:
            self.text_txt = self.train_text_txt
        else:
            self.text_txt = self.test_text_txt

        self.classnames = self.read_classnames()
        self.names = [self.classnames[label] for label in self.labels]

        text_by_photo = self.read_text_by_photoid(self.text_txt)
        self.texts = []
        for path in self.img_path:
            photo_id = os.path.splitext(os.path.basename(path))[0]
            self.texts.append(text_by_photo.get(photo_id, ""))

        self._drop_missing_captions(split_name)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        name = self.names[index]
        text = self.texts[index]
        return image, label, name, text

    @staticmethod
    def read_text_by_photoid(text_txt):
        text_by_photo = {}
        with open(text_txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if not parts:
                    continue
                photo_id = parts[0]
                caption = parts[1] if len(parts) > 1 else ""
                text_by_photo[photo_id] = caption
        return text_by_photo

    @classmethod
    def read_classnames(self):
        classnames = []
        with open(self.classnames_txt, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                if not parts:
                    continue
                classname = " ".join(parts[1:])
                classnames.append(classname)
        return classnames

    def _drop_missing_captions(self, split_name):
        before_labels = list(self.labels)
        kept_idxs = [i for i, text in enumerate(self.texts) if isinstance(text, str) and text.strip() != ""]
        num_before = len(self.labels)
        num_after = len(kept_idxs)

        self.img_path = [self.img_path[i] for i in kept_idxs]
        self.labels = [self.labels[i] for i in kept_idxs]
        self.names = [self.names[i] for i in kept_idxs]
        self.texts = [self.texts[i] for i in kept_idxs]

        before_counter = Counter(before_labels)
        label_counter = Counter(self.labels)
        self.num_classes = len(self.classnames)
        self.cls_num_list = [label_counter.get(i, 0) for i in range(self.num_classes)]
        cls_num_list_before = [before_counter.get(i, 0) for i in range(self.num_classes)]
        cls_num_delta = [self.cls_num_list[i] - cls_num_list_before[i] for i in range(self.num_classes)]

        self.filter_stats = {
            "split": split_name,
            "num_before": num_before,
            "num_after": num_after,
            "num_dropped": num_before - num_after,
            "cls_num_list_before": cls_num_list_before,
            "cls_num_list_after": list(self.cls_num_list),
            "cls_num_delta": cls_num_delta,
        }

        dropped = num_before - num_after
        print(f"Yelp_MM_LT[{split_name}] drop empty caption: {dropped} (from {num_before} to {num_after})")
