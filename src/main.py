"""Main script for training, validation, and inference."""

# Imports Python builtins.
from copy import deepcopy
import os
import os.path as osp
import resource

# Imports PyTorch packages.
import torch

# Imports other packages.
from configargparse import Parser
from PIL import ImageFile
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.utilities.seed import seed_everything

# Imports local packages.
from args import parse_args
from coco_tools.coco import coco_loader
from coco_tools.infer import infer_loader
from model.ws_detr import WS_DETR
from utils.misc import get_state_dict_from_checkpoint

# Prevents PIL from throwing invalid error on large image files.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Prevents DataLoader memory error.
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


def load_class_agnostic_detector(args):
    """Loads class agnostic detector and freezes parameters."""

    agnostic_args = deepcopy(args)
    agnostic_args.classes = 2
    class_agnostic_detector = WS_DETR(agnostic_args)

    checkpoint = torch.load(args.class_agnostic_weights, map_location="cpu")
    state_dict = get_state_dict_from_checkpoint(checkpoint)
    class_agnostic_detector.load_state_dict(state_dict, strict=False)
    print(
        f"Class-agnostic detector loaded from {args.class_agnostic_weights}."
    )

    class_agnostic_detector.eval()
    for p in class_agnostic_detector.parameters():
        p.requires_grad = False

    return class_agnostic_detector

def load_model(args, coco_groundtruth, class_names=None):
    """Loads WS-DETR model and optionally loads weights."""

    # Loads class-agnostic detector during training.
    # Otherwise, it is saved in the WS-DETR weights.
    class_agnostic_detector = None
    if args.task == "train" or args.supervised:
        class_agnostic_detector = load_class_agnostic_detector(args)

    # Instantiates WS-DETR model.
    model = WS_DETR(
        args,
        class_agnostic_detector=class_agnostic_detector,
        class_names=class_names,
        coco_groundtruth=coco_groundtruth,
    )

    # Loads model weights.
    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu")
        state_dict = get_state_dict_from_checkpoint(checkpoint)

        if args.resume_training and args.weights.endswith("ckpt"):
            args.ckpt_path = args.weights
            print(f"Resuming training state from {args.weights}.")
        elif args.resume_weights or args.task in ("test", "infer"):
            model.load_state_dict(state_dict, strict=False)
            print(f"Weights loaded from {args.weights}.")
        else:
            # Drops MIL head from checkpoint.
            state_dict = {
                k: v for k, v in state_dict.items()
                if "class_embed" not in k
                and "det_embed" not in k
            }

            model.load_state_dict(state_dict, strict=False)
            print(f"Weights loaded from {args.weights}.")

    return model

def load_trainer(args):
    """Loads PyTorch Lightning Trainer with callbacks."""

    # Instantiates checkpointer to save model
    # when a new best mAP is reached.
    checkpointer = ModelCheckpoint(
        filename="{epoch}-{mAP:.2f}",
        mode="max",
        monitor="mAP",
        save_last=True,
    )

    # Instantiates progress bar. Changing refresh rate is useful when
    # stdout goes to a logfile (e.g., on cluster). 1 is normal and 0 disables.
    progress_bar = TQDMProgressBar(refresh_rate=args.refresh_rate)

    # Sets DDP strategy for multi-GPU training.
    args.strategy = "ddp" if args.gpus > 1 else None

    # Instantiates PL Trainer using args.
    callbacks = [checkpointer, progress_bar]
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks)

    return trainer

def main(args):
    """Trains, tests, or infers with model as specified by args."""

    # Sets global seed for reproducibility.
    # Note: Due to CUDA operations which cannot be made deterministic,
    # the code will still not be perfectly reproducible.
    seed_everything(seed=42, workers=True)

    # Sets output directory.
    if "PT_OUTPUT_DIR" in os.environ:
        args.default_root_dir = os.environ["PT_OUTPUT_DIR"]
    elif args.save_dir:
        args.default_root_dir = args.save_dir
    else:
        args.default_root_dir = os.getcwd()

    # Sets outputs directory for inference images.
    args.imgs_dir = osp.join(args.default_root_dir, "imgs")
    os.makedirs(args.imgs_dir, exist_ok=True)

    # Instantiates COCO dataloaders and ground truth.
    if args.task == "train":
        train_loader = coco_loader(args, task="train")
        val_loader = coco_loader(args, task="val")
        coco_groundtruth = val_loader.dataset.coco
    elif args.task == "test":
        val_loader = coco_loader(args, task="test")
        coco_groundtruth = val_loader.dataset.coco
    elif args.task == "infer":
        val_loader = infer_loader(args)
        coco_groundtruth = None

    class_names = None
    if args.task != "infer":
        cats = list(val_loader.dataset.coco.cats.values())
        cats = sorted(cats, key=lambda x: x["id"])
        class_names = [c["name"] for c in cats]
        
    model = load_model(args, coco_groundtruth, class_names=class_names)
    trainer = load_trainer(args)
        
    if args.task == "train":
        trainer.fit(model, train_loader, val_loader)
    elif args.task == "test":
        trainer.test(model, val_loader)
    elif args.task == "infer":
        trainer.predict(model.eval(), val_loader)
       

if __name__ == "__main__":
    args = parse_args()

    main(args)

