"""Sets command line and config file arguments."""

# Imports other packages.
from configargparse import Parser
from pytorch_lightning import Trainer

# Imports local packages.
from model.ws_detr import WS_DETR


def parse_args():
    """Parses command line and config file arguments."""

    # Instantiates config arg parser with required config file.
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    # Adds command line, Trainer, and model arguments.
    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser = WS_DETR.add_model_specific_args(parser)

    args = parser.parse_args()

    return args

def add_input_args(parser):
    """Adds arguments not handled by Trainer or model."""

    parser.add(
        "--train_imgs_dir",
        help="Training images directory.",
    )
    parser.add(
        "--train_anns",
        help="Training labels formatted as COCO json.",
    )
    parser.add(
        "--val_imgs_dir",
        help="Validation images directory.",
    )
    parser.add(
        "--val_anns",
        help="Validation labels formatted as COCO json.",
    )
    parser.add(
        "--test_imgs_dir",
        help="Testing images directory.",
    )
    parser.add(
        "--test_anns",
        help="Testing labels formatted as COCO json.",
    )
    parser.add(
        "--infer_imgs_dir",
        help="Images directory for performing inference.",
    )
    parser.add(
        "--save_dir",
        default="out",
        help="Directory to save images and checkpoints; overrides PT dir.",
    )

    parser.add(
        "--task",
        choices=["train", "test", "infer"],
        help="Mode to run the model in.",
    )

    parser.add(
        "--batch_size",
        type=int,
        help="Number of images per batch.",
    )
    parser.add(
        "--class_agnostic_weights",
        help="Filepath of class-agnostic model weights.",
    )
    parser.add(
        "--classes",
        type=int,
        help="Number of classes in the dataset.",
    )
    parser.add(
        "--dropout",
        type=float,
        help="Dropout probability in the Transformer.",
    )
    parser.add(
        "--joint_probability",
        action="store_true",
        help=(
            "Whether to use our joint probability technique instead of"
            " learning the detection branch in the MIL classifier."
        ),
    )
    parser.add(
        "--infer_display_thresh",
        type=float,
        help="Confidence threshold to display images during inference.",
    )
    parser.add(
        "--nms_thresh",
        type=float,
        help="IoU threshold for non-maximum suppression (0 for no NMS).",
    )
    parser.add(
        "--offset",
        type=int,
        help="Offset of image label indices.",
    )
    parser.add(
        "--refresh_rate",
        type=int,
        help="Batch interval for updating training progress bar.",
    )
    parser.add(
        "--resume_training",
        action="store_true",
        help="Whether to resume training using the PL Trainer.",
    )
    parser.add(
        "--resume_weights",
        action="store_true",
        help="Whether to load all possible model weights from checkpoint.",
    )
    parser.add(
        "--sampler",
        action="store_true",
        help="Whether to use a balanced random sampler in DataLoader.",
    )
    parser.add(
        "--sparse",
        action="store_true",
        help=(
            "Whether to use sparsemax instead of softmax in the MIL head"
            " across the detections dimension."
        ),
    )
    parser.add(
        "--supervised",
        action="store_true",
        help=(
            "Whether a fully-supervised model is being used for testing or"
            " inference (e.g., when visualizing class-agnostic boxes)."
        ),
    )
    parser.add(
        "--viz_test_batches",
        type=int,
        help=("How many batches to visualize with prediction and"
              " ground-truth boxes during validation and test steps."
        ),
    )
    parser.add(
        "--weights",
        help="Filepath of model weights.",
    )
    parser.add(
        "--workers",
        type=int,
        help="Number of workers in DataLoader.",
    )

    parser.add(
        "--lr_backbone",
        type=float,
        help="Learning rate for backbone and input projection.",
    )
    parser.add(
        "--lr_detr",
        type=float,
        help="Learning rate for DETR modules.",
    )
    parser.add(
        "--lr_drop",
        type=float,
        help=("Factor by which to drop the learning rate every"
              " lr_patience epochs with no loss improvement."
        ),
    )
    parser.add(
        "--lr_mil",
        type=float,
        help="Learning rate for MIL head."
    )
    parser.add(
        "--lr_patience",
        type=int,
        help=("How many epochs with no loss improvement"
              " after which the learning rate will drop."
        ),
    )
    parser.add(
        "--lr_step_size",
        type=int,
        help="How many epochs to run before dropping the learning rate.",
    )
    parser.add(
        "--objectness_scale",
        type=float,
        help="Scaling term for objectness regularization.",
    )
    parser.add(
        "--weight_decay",
        type=float,
        help="Weight decay factor.",
    )

    return parser

