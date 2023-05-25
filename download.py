"""Downloads datasets and creates train/test splits."""

# Imports Python builtins.
from copy import deepcopy
import json
import math
import os
import os.path as osp
import random
import requests
import shutil

# Imports other packages.
from configargparse import Parser
import gdown
from PIL import Image


def load_anns(path):
    """Loads annotation dict from JSON file."""

    with open(path, "r") as f:
       anns = json.load(f)
    return anns

def save_anns(anns, path):
    """Saves annotation dict to disk as JSON file."""

    with open(path, "w") as f:
        json.dump(anns, f)

def make_class_agnostic(anns):
    """Sets all categories to 1 in an annotation dict."""

    class_agnostic_anns = deepcopy(anns)

    categories = [
        {"id": 0, "name": "no object"},
        {"id": 1, "name": "object"},
    ]

    class_agnostic_anns["categories"] = categories

    for ann in class_agnostic_anns["annotations"]:
        ann["category_id"] = 1

    return class_agnostic_anns

def make_subset(anns, img_ids):
    """Gets images corresponding to ids from an annotation dict."""

    subset_anns = deepcopy(anns)

    subset_anns["images"] = [
        img for img in anns["images"] if img["id"] in img_ids
    ]
    subset_anns["annotations"] = [
        ann for ann in anns["annotations"] if ann["image_id"] in img_ids
    ]

    return subset_anns

def get_classes_by_img(anns):
    """Gets dict of {img: classes}."""

    classes_by_img = {img["id"]: [] for img in anns["images"]}
    for ann in anns["annotations"]:
        classes_by_img[ann["image_id"]].append(ann["category_id"])
    classes_by_img = {
        img_id: set(classes) for img_id, classes in classes_by_img.items()
    }

    return classes_by_img

def get_num_samples_by_cls(anns, img_ids=None):
    """Gets dict of {cls: num samples}."""

    if img_ids:
        img_ids = set(img_ids)
    else:
        img_ids = set([img["id"] for img in anns["images"]])

    num_samples_by_cls = {}
    for ann in anns["annotations"]:
        if ann["image_id"] in img_ids:
            if ann["category_id"] in num_samples_by_cls:
                num_samples_by_cls[ann["category_id"]] += 1
            else:
                num_samples_by_cls[ann["category_id"]] = 1

    return num_samples_by_cls

def make_splits(path, train_pcts, splits=1):
    """Makes random train/test split(s) from an annotation file."""
    
    anns = load_anns(path)
    img_ids = [img["id"] for img in anns["images"]]

    for train_pct in train_pcts:
        for seed in range(splits):
            random.seed(seed)
            train_size = math.ceil(train_pct * len(img_ids))

            train_img_ids = random.sample(img_ids, train_size)
            test_img_ids = [
                img_id for img_id in img_ids if img_id not in train_img_ids
            ]

            x = zip(("train", "test"), (train_img_ids, test_img_ids))
            for name, ids in x:
                subset_anns = make_subset(anns, ids)
                class_agnostic_anns = make_class_agnostic(subset_anns)

                suffix = f"_{name}_seed{seed}_trn{train_pct}.json"
                subset_anns_path = osp.splitext(path)[0] + suffix
                class_agnostic_anns_path = osp.splitext(subset_anns_path)[0] \
                    + "_class_agnostic.json"

                save_anns(subset_anns, subset_anns_path)
                save_anns(class_agnostic_anns, class_agnostic_anns_path)

def make_cls_splits(path, cls_pcts, splits=1):
    """Makes random class-wise train/test splits from an annotation file."""

    anns = load_anns(path)
    img_ids = [img["id"] for img in anns["images"]]
    classes = [cls["id"] for cls in anns["categories"]]
    classes_by_img = get_classes_by_img(anns)

    for cls_pct in cls_pcts:
        cls_size = math.ceil(cls_pct * len(classes))

        for seed in range(splits):
            random.seed(seed)
            cls_ids = random.sample(classes, cls_size)

            train_img_ids = []
            for img_id in img_ids:
                if all(cls in cls_ids for cls in classes_by_img[img_id]):
                    train_img_ids.append(img_id)

            subset_anns = make_subset(anns, train_img_ids)
            class_agnostic_anns = make_class_agnostic(subset_anns)

            suffix = f"_train_seed{seed}_cls{cls_pct}.json"
            subset_anns_path = osp.splitext(path)[0] + suffix
            class_agnostic_anns_path = osp.splitext(subset_anns_path)[0] \
                + "_class_agnostic.json"

            save_anns(subset_anns, subset_anns_path)
            save_anns(class_agnostic_anns, class_agnostic_anns_path)

def fgvc_download(base_dir):
    """Downloads and extracts FGVC dataset."""

    print("Downloading FGVC-Aircraft dataset...")

    url = (
        "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/"
        "archives/fgvc-aircraft-2013b.tar.gz"
    )

    fgvc_dir = osp.join(base_dir, "fgvc-aircraft-2013b")
    os.makedirs(fgvc_dir, exist_ok=True)

    # Downloads FGVC dataset from VGG.
    data = requests.get(url)
    tar_path = osp.join(base_dir, "fgvc-aircraft-2013b.tar.gz")
    with open(tar_path, "wb") as f:
        f.write(data.content)

    print("Done.")
    print("Extracting FGVC-Aircraft dataset...")

    # Extracts FGVC dataset. May take a while.
    tmp_dir = osp.join(base_dir, "tmp")
    shutil.unpack_archive(tar_path, tmp_dir)

    print("Done.")
    print("Formatting images...")

    tmp_fgvc_dir = osp.join(tmp_dir, "fgvc-aircraft-2013b", "data")
    old_imgs_dir = osp.join(tmp_fgvc_dir, "images")
    imgs_dir = osp.join(fgvc_dir, "images")
    shutil.move(old_imgs_dir, imgs_dir)

    # Crops images (the bottom 20px are an info banner).
    name_to_wh = {}
    for img_name in os.listdir(imgs_dir):
        img_path = osp.join(imgs_dir, img_name)
        img = Image.open(img_path)
        w, h = img.size

        img = img.crop((0, 0, w, h - 20))
        img.save(img_path)

        name_to_wh[osp.splitext(img_name)[0]] = img.size

    print("Done.")
    print("Converting annotations...")

    # Gets FGVC classes.
    name_to_id = {}
    categories = []
    fgvc_cat_path = osp.join(tmp_fgvc_dir, "variants.txt")
    with open(fgvc_cat_path, "r") as f:
        for j, line in enumerate(f):
            line = line.strip("\n")
            name_to_id[line] = j
            categories.append({"id": j, "name": line})

    # Gets FGVC boxes.
    id_to_box = {}
    fgvc_box_path = osp.join(tmp_fgvc_dir, "images_box.txt")
    with open(fgvc_box_path, "r") as f:
        for line in f:
            box = line.strip("\n").split()
            id_to_box[box[0]] = [int(b) for b in box[1:]]

    # Makes COCO json annotations.
    ann_dir = osp.join(fgvc_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    for split in ("trainval", "test"):
        ann = {"images": [], "categories": categories, "annotations": []}

        fgvc_ann_path = osp.join(tmp_fgvc_dir, f"images_variant_{split}.txt")
        with open(fgvc_ann_path, "r") as f:
            for j, line in enumerate(f):
                line = line.strip("\n").split()
                name = f"{line[0]}.jpg"
                cat = name_to_id[" ".join(line[1:])]
                box = id_to_box[line[0]]
                wh = name_to_wh[line[0]]

                img = {
                    "file_name": name,
                    "height": wh[1],
                    "id": j,
                    "width": wh[0],
                }

                box_ann = {
                    "area": wh[0] * wh[1],
                    "bbox": box,
                    "category_id": cat,
                    "id": j,
                    "image_id": j,
                    "iscrowd": 0,
                }

                ann["images"].append(img)
                ann["annotations"].append(box_ann)

        json_path = osp.join(ann_dir, f"{split}.json")
        with open(json_path, "w") as f:
            json.dump(ann, f)

    shutil.rmtree(tmp_dir)

    print("Done.")

def fsod_download(base_dir):
    """Downloads and extracts FSOD dataset."""

    url = (
        "https://drive.google.com/drive/folders/"
        "1XXADD7GvW8M_xzgFpHfudYDYtKtDgZGM"
    )

    # Downloads FSOD dataset from Google Drive.
    fsod_dir = osp.join(base_dir, "fsod")
    gdown.download_folder(url, output=fsod_dir)

    print("Extracting FSOD dataset...")

    # Extracts FSOD dataset. May take a while.
    img_dir = osp.join(fsod_dir, "images")
    for img_tar_name in os.listdir(img_dir):
        img_tar_path = osp.join(img_dir, img_tar_name)
        out_path = osp.splitext(img_tar_path)[0]
        shutil.unpack_archive(img_tar_path, out_path)

    print("Done.")

def fsod_split(base_dir):
    """Creates train/test splits from FSOD dataset."""

    print("Creating FSOD splits...")

    ann_dir = osp.join(base_dir, "fsod", "annotations")

    fsod_train_anns_path = osp.join(ann_dir, "fsod_train.json")
    fsod_800_anns_path = osp.join(ann_dir, "fsod_800.json")
    fsod_test_anns_path = osp.join(ann_dir, "fsod_test.json")
    fsod_200_anns_path = osp.join(ann_dir, "fsod_200.json")

    # Renames base annotations.
    shutil.copyfile(fsod_train_anns_path, fsod_800_anns_path)
    shutil.copyfile(fsod_test_anns_path, fsod_200_anns_path)

    # Saves class-agnostic annotations.
    for anns_path in (fsod_800_anns_path, fsod_200_anns_path):
        anns = load_anns(anns_path)
        class_agnostic_anns_path = anns_path[:-5] + "_class_agnostic.json"
        class_agnostic_anns = make_class_agnostic(anns)
        save_anns(class_agnostic_anns, class_agnostic_anns_path)

    # Creates 3x 80/20 train/test splits from FSOD 200.
    make_splits(fsod_200_anns_path, [0.8], splits=3)

    # Makes 20/40/60/80 class and data splits from FSOD 800.
    make_cls_splits(fsod_800_anns_path, [0.2, 0.4, 0.6, 0.8], splits=3)
    make_splits(fsod_800_anns_path, [0.2, 0.4, 0.6, 0.8], splits=3)

    print("Done.")

def inaturalist_download(base_dir):
    """Downloads and extracts iNaturalist 2017 dataset."""

    print("Downloading iNaturalist dataset...")

    base_url = "https://ml-inat-competition-datasets.s3.amazonaws.com/2017/"
    img_url = osp.join(base_url, "train_val_images.tar.gz")
    train_anns_url = osp.join(base_url, "train_2017_bboxes.zip")
    val_anns_url = osp.join(base_url, "val_2017_bboxes.zip")

    # Downloads iNaturalist 2017 images from AWS.
    img_data = requests.get(img_url)
    img_tar_path = osp.join(base_dir, "inaturalist/train_val_images.tar.gz")
    with open(img_tar_path, "wb") as f:
        f.write(img_data.content)

    train_anns_data = requests.get(train_anns_url)
    train_anns_zip_path = osp.join(base_dir, "inaturalist/train_2017_bboxes.zip")
    with open(train_anns_zip_path, "wb") as f:
        f.write(train_anns_data.content)

    val_anns_data = requests.get(val_anns_url)
    val_anns_zip_path = osp.join(base_dir, "inaturalist/val_2017_bboxes.zip")
    with open(val_anns_zip_path, "wb") as f:
        f.write(val_anns_data.content)

    print("Done.")
    print("Extracting iNaturalist dataset...")

    inat_path = osp.join(base_dir, "inaturalist")
    anns_path = osp.join(inat_path, "annotations")

    # Extracts iNaturalist 2017 images and annotations. May take a while.
    shutil.unpack_archive(img_tar_path, inat_path)
    shutil.unpack_archive(train_anns_zip_path, anns_path)
    shutil.unpack_archive(val_anns_zip_path, anns_path)

    print("Done.")
    print("Formatting iNaturalist annotations...")

    train_anns = json.load(open(osp.join(anns_path, "train_2017_bboxes.json"), "r"))
    val_anns = json.load(open(osp.join(anns_path, "val_2017_bboxes.json"), "r"))

    # Makes clean fine-grained annotation files.
    id_to_name = {c["id"]: c["name"] for c in train_anns["categories"]}
    ann_cat_ids = [ann["category_id"] for ann in train_anns["annotations"]]
    class_names = [id_to_name[ann["category_id"]] for ann in train_anns["annotations"]]
    class_names = sorted(list(set(class_names)))
    classes = [{"id": j, "name": n} for j, n in enumerate(class_names)]
    name_to_id = {c["name"]: c["id"] for c in classes}
    old_to_new = {c["id"]: name_to_id[c["name"]] for c in train_anns["categories"]}

    for split, anns in zip(("train", "val"), (train_anns, val_anns)):
        clean_anns = deepcopy(anns)
        clean_anns["categories"] = classes
        for ann in clean_anns["annotations"]:
            ann["category_id"] = old_to_new[ann["category_id"]]

        clean_anns_path = osp.join(anns_path, f"{split}_2017_bboxes_clean.json")
        with open(clean_anns_path, "w") as f:
            json.dump(clean_anns, f)

    # Makes superclass annotation files.
    superclass_names = sorted(list(set([c["supercategory"] for c in train_anns["categories"]])))
    superclasses = [{"id": j, "name": n} for j, n in enumerate(superclass_names)]
    name_to_id = {c["name"]: c["id"] for c in superclasses}
    old_to_new = {c["id"]: name_to_id[c["supercategory"]] for c in train_anns["categories"]}
    
    for split, anns in zip(("train", "val"), (train_anns, val_anns)):
        superclass_anns = deepcopy(anns)
        superclass_anns["categories"] = superclasses
        for ann in superclass_anns["annotations"]:
            ann["category_id"] = old_to_new[ann["category_id"]]

        superclass_anns_path = osp.join(anns_path, f"{split}_2017_bboxes_superclass.json")
        with open(superclass_anns_path, "w") as f:
            json.dump(superclass_anns, f)
    
    print("Done.")

if __name__ == "__main__":
    parser = Parser()

    parser.add(
        "--base_dir",
        default="data",
        help="Where to extract dataset files.",
    )
    parser.add(
        "--datasets",
        choices=["all", "fgvc", "fsod", "inaturalist"],
        default="all",
        nargs="+",
        help="Which dataset(s) to download.",
    )

    args = parser.parse_args()
    if "all" in args.datasets:
        args.datasets = ["fgvc", "fsod", "inaturalist"]

    os.makedirs(args.base_dir, exist_ok=True)

    if "fgvc" in args.datasets:
        fgvc_download(args.base_dir)
    if "fsod" in args.datasets:
        fsod_download(args.base_dir)
        fsod_split(args.base_dir)
    if "inaturalist" in args.datasets:
        inaturalist_download(args.base_dir)

