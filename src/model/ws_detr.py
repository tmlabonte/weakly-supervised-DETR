"""Defines WS-DETR LightningModule."""

# Imports Python builtins.
from io import StringIO
import math
import sys

# Imports PyTorch packages.
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torchvision.transforms as T

# Imports other packages.
from azureml.core.run import Run
import pytorch_lightning as pl

# Imports local packages.
from coco_tools.coco import coco_evaluate
from utils.box_ops import box_cxcywh_to_xyxy
from utils.misc import (
    compute_accuracy,
    exclude_params,
    gather_coco_results_across_gpus,
    inverse_sigmoid,
    NestedTensor,
    nested_tensor_from_tensor_list,
    save_infer_img,
)

# Imports local model packages.
from .backbone import build_backbone
from .deformable_transformer import DeformableTransformer
from .mil_loss import mil_loss
from .mlp import MLP
from .model_args import add_model_args
from .postprocess import postprocess


class WS_DETR(pl.LightningModule):
    """Defines Weakly Supervised Detection Transformer (WS-DETR)."""

    def __init__(
        self,
        args,
        class_agnostic_detector=None,
        class_names=None,
        coco_groundtruth=None,
    ):
        """Initializes WS-DETR with backbone, Transformer, and embeddings."""

        super().__init__()

        # Saves class names for visualization.
        if args.classes == 2:
            args.offset = 0
            self.class_names = ["no object", "object"]
        else:
            self.class_names = class_names

        # Saves hyperparameters to self.hparams.
        self.save_hyperparameters(args)
        
        # Saves instance variables.
        self.class_agnostic_detector = class_agnostic_detector
        self.coco_groundtruth = coco_groundtruth
        self.queries = args.queries
        self.feature_levels = args.feature_levels

        # Initializes backbone and Transformer.
        self.backbone = build_backbone(args)
        self.transformer = DeformableTransformer(args)
        self.query_embed = nn.Embedding(args.queries, args.hidden_dim * 2)

        # Initializes box and MIL head embeddings.
        self.bbox_embed = MLP(args.hidden_dim, args.hidden_dim, 4, 3)
        self.det_embed = nn.Linear(args.hidden_dim, args.classes)
        self.class_embed = nn.Linear(args.hidden_dim, args.classes)

        # Initializes input projection.
        self.init_input_proj(args.feature_levels, args.hidden_dim)

        # Initializes weights and biases for embeddings.
        self.init_weights_and_biases(args.classes)
        
        # Duplicates embeddings for each decoder layer.
        layers = range(self.transformer.decoder.num_layers)
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in layers])
        self.det_embed = nn.ModuleList([self.det_embed for _ in layers])
        self.class_embed = nn.ModuleList([self.class_embed for _ in layers])
        
        # Shares class-agnostic query embedding with multi-class model.
        if class_agnostic_detector:
            self.query_embed = class_agnostic_detector.query_embed

        # Freezes query embedding.
        for p in self.query_embed.parameters():
            p.requires_grad = False
            
    def init_input_proj(self, feature_levels, hidden_dim):
        """Initializes input projection."""

        if feature_levels > 1:
            # Initializes multi-scale input projection.
            num_backbone_outs = len(self.backbone.strides)
            input_proj_list = []

            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))

            for _ in range(feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            # Initializes single-scale input projection.
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        self.backbone.num_channels[0],
                        hidden_dim,
                        kernel_size=1,
                    ),
                    nn.GroupNorm(32, hidden_dim),
                )])

    def init_weights_and_biases(self, classes):
        """Initializes embedding weights and biases."""

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        # Initializes MIL head embeddings.
        self.det_embed.bias.data = torch.ones(classes) * bias_value
        self.class_embed.bias.data = torch.ones(classes) * bias_value

        # Initializes bbox embeddings.
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)

        # Initializes input projection.
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, imgs):
        """Applies WS-DETR to a batch of images."""

        # Casts images to NestedTensor.
        if not isinstance(imgs, NestedTensor):
            imgs = nested_tensor_from_tensor_list(imgs)

        # Computes agnostic boxes and object confidences.
        if self.class_agnostic_detector:
            agnostic_outputs = self.class_agnostic_detector(imgs)
            agnostic_boxes = agnostic_outputs["boxes"]
            agnostic_scores = agnostic_outputs["classes_logits"].sigmoid()
            obj_confs = agnostic_scores[:, :, 1]
        
        # Extracts features and position embedding using backbone.
        features, pos = self.backbone(imgs)

        # Decomposes NestedTensor and feeds image through input projection.
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        # Computes multi-scale feature maps.
        if self.feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = imgs.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:])
                mask = mask.to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # Passes images, masks, and position embedding through Transformer
        # with query embedding applied in the decoder.
        hs, init_reference, inter_references = self.transformer(
            srcs,
            masks,
            pos,
            self.query_embed.weight,
        )

        # Computes multi-level embedding outputs from Transformer embedding.
        boxes = []
        classes = []
        dets = []
        for lvl in range(hs.shape[0]):
            lvl_boxes = self.bbox_embed[lvl](hs[lvl])
            lvl_classes = self.class_embed[lvl](hs[lvl])
            lvl_dets = self.det_embed[lvl](hs[lvl])

            # Postprocesses box embedding with reference points.
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            if reference.shape[-1] == 4:
                lvl_boxes += reference
            else:
                assert reference.shape[-1] == 2
                lvl_boxes[..., :2] += reference

            lvl_boxes = lvl_boxes.sigmoid()
            boxes.append(lvl_boxes)
            classes.append(lvl_classes)
            dets.append(lvl_dets)

        # Converts lists to tensors.
        boxes = torch.stack(boxes)
        classes = torch.stack(classes)
        dets = torch.stack(dets)

        # Combines results into output dict.
        out = {
            "boxes": boxes[-1],
            "classes_logits": classes[-1],
            "dets_logits": dets[-1],
        }

        # Replaces box prediction with class agnostic boxes and
        # adds objectness confidences to results.
        if self.class_agnostic_detector:
            out["boxes"] = agnostic_boxes
            out["obj_confs"] = obj_confs
        
        return out

    def configure_optimizers(self):
        """Configures AdamW optimizer and StepLR scheduler."""

        # Separates backbone, DETR, and MIL parameters.
        backbone_params = list(self.backbone.parameters())
        backbone_params.extend(list(self.input_proj.parameters()))

        mil_params = list(self.class_embed.parameters())
        mil_params.extend(list(self.det_embed.parameters()))

        to_exclude = [backbone_params, mil_params]
        detr_params = exclude_params(
            self.parameters(),
            to_exclude,
        )

        # Assigns different learning rates to backbone, DETR, and MIL head.
        param_dicts = [
            {
                "params": mil_params,
                "lr": self.hparams.lr_mil,
            },
            {
                "params": detr_params,
                "lr": self.hparams.lr_detr,
            },
            {
                "params": backbone_params,
                "lr": self.hparams.lr_backbone,
            },
        ]

        # Initializes AdamW optimizer with specified LR and weight decay.
        optimizer = AdamW(
            param_dicts,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_patience and self.hparams.lr_step_size:
            raise ValueError(
                "Please only enable one of lr_patience (for ReduceLROnPlateau"
                " scheduler) and lr_step_size (for StepLR scheduler)."
            )

        # Initializes scheduler which drops LR by a factor of 10
        # if it does not decrease within lr_patience epochs.
        if self.hparams.lr_patience:
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=self.hparams.lr_patience,
            )
        # Initializes scheduler which drops LR by a factor of 10
        # after every lr_step_size epochs.
        elif self.hparams.lr_step_size:
            scheduler = StepLR(
                optimizer,
                self.hparams.lr_step_size,
                gamma=self.hparams.lr_drop,
            )

        # Builds optimizer config as expected by PL.
        cfg = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

        return cfg

    def forward_with_loss(self, batch, idx):
        """Computes prediction and loss."""

        imgs, targets = batch

        outputs = self(imgs)

        # Computes MIL and objectness loss.
        mil, obj = mil_loss(
            outputs,
            targets,
            joint_probability=self.hparams.joint_probability,
            objectness_scale=self.hparams.objectness_scale,
            offset=self.hparams.offset,
            sparse=self.hparams.sparse,
        )

        loss = torch.stack((mil, obj)).sum()

        return outputs, loss

    def training_step(self, batch, idx):
        """Computes loss."""

        _, loss = self.forward_with_loss(batch, idx)

        return loss

    def training_epoch_end(self, training_step_outputs):
        """Computes and logs epoch training loss."""

        # Gathers loss across GPUs.
        loss = torch.stack(training_step_outputs).mean()
        loss = self.all_gather(loss).mean.item()

        if self.trainer.is_global_zero:
            try:
                # Logs to AzureML.
                writer = Run.get_context(allow_offline=False)
                writer.log("Train Loss", loss)
            except:
                pass

    def validation_step(self, batch, idx):
        """Computes loss and postprocesses prediction."""

        imgs, targets = batch
        orig_sizes = torch.stack([t["orig_size"] for t in targets])

        # Plots predictions with ground-truth boxes.
        if idx < self.hparams.viz_test_batches:
            names = [str(t["image_id"].item()) + ".jpg" for t in targets]
            self.predict_helper(imgs, names, orig_sizes, targets=targets)

        outputs, loss = self.forward_with_loss(batch, idx)

        # Postprocesses model outputs for COCO metrics computation.
        results = postprocess(
            outputs,
            orig_sizes,
            joint_probability=self.hparams.joint_probability,
            nms_thresh=self.hparams.nms_thresh,
            offset=self.hparams.offset,
            sparse=self.hparams.sparse,
            supervised=self.hparams.supervised,
        )

        for target, result in zip(targets, results):
            result["image_id"] = target["image_id"]

        return results, loss

    def validation_epoch_end(self, validation_step_outputs):
        """Computes COCO metrics over all validation batches."""

        results = []
        losses = []
        for result, loss in validation_step_outputs:
            results.extend(result)
            losses.append(loss)

        # Gathers loss across GPUs.
        loss = torch.stack(losses).mean()
        loss = self.all_gather(loss).mean().item()

        # Gathers COCO results across GPUs.
        results = self.all_gather(results)
        coco_results = gather_coco_results_across_gpus(results)

        # Performs COCO evaluation while suppressing prints so
        # it doesn't print on every GPU. I tried doing evaluation
        # on rank zero only, but it doesn't work (possible PL bug).
        coco_prints = StringIO()
        sys.stdout = coco_prints
        stats = coco_evaluate(coco_results, self.coco_groundtruth)
        sys.stdout = sys.__stdout__
        top1_acc, top5_acc = compute_accuracy(
            coco_results,
            self.coco_groundtruth,
        )

        # Logs to PL logger and syncs across GPUs.
        self.log("val_loss", loss, sync_dist=True)
        self.log("mAP", stats[0], sync_dist=True)
        self.log("AP50", stats[1], sync_dist=True)
        self.log("AP75", stats[2], sync_dist=True)
        self.log("Top1 Acc", top1_acc, sync_dist=True)
        self.log("Top5 Acc", top5_acc, sync_dist=True)

        if self.trainer.is_global_zero:
            try:
                # Prints COCO evaluation results.
                print(coco_prints.getvalue().strip("\n"))

                # Logs to AzureML.
                writer = Run.get_context(allow_offline=False)
                writer.log("Val Loss", loss)
                writer.log("mAP", stats[0])
                writer.log("AP50", stats[1])
                writer.log("AP75", stats[2])
                writer.log("Top1 Acc", top1_acc)
                writer.log("Top5 Acc", top5_acc)
            except:
                pass

    def test_step(self, batch, idx):
        """Computes loss and postprocesses prediction."""
        
        return self.validation_step(batch, idx)

    def test_epoch_end(self, test_step_outputs):
        """Computes COCO metrics over all test batches."""

        return self.validation_epoch_end(test_step_outputs)

    def predict_helper(self, imgs, names, orig_sizes, targets=None):
        """Helper function for prediction and image saving."""

        outputs = self(imgs)

        results = postprocess(
            outputs,
            orig_sizes,
            joint_probability=self.hparams.joint_probability,
            nms_thresh=self.hparams.nms_thresh,
            offset=self.hparams.offset,
            sparse=self.hparams.sparse,
            supervised=self.hparams.supervised,
        )

        if not targets:
            targets = [None for _ in range(len(results))]

        z = zip(imgs.tensors, imgs.mask, names, orig_sizes, results, targets)
        for img, mask, name, orig_size, result, target in z:
            # Applies mask to get original tensor.
            for j, x in enumerate(mask[0]):
                if x:
                    break
            for i, x in enumerate(torch.transpose(mask, 0, 1)[0]):
                if x:
                    break
            img = img[:, :i, :j]
            img = T.Resize(orig_size.int().tolist())(img)

            # Thresholds output for display.
            keep = result["confs"] > self.hparams.infer_display_thresh
            boxes = result["boxes"][keep]
            confs = result["confs"][keep]
            preds = result["preds"][keep] 

            target_boxes = None if not target else target["boxes"]
            target_labels = None if not target else target["box_labels"]

            if target_boxes is not None:
                target_boxes = box_cxcywh_to_xyxy(target_boxes)

                # Converts from [0, 1] to [0, height] coordinates.
                img_h, img_w = orig_size
                dims = [img_w, img_h, img_w, img_h]
                scale_fct = torch.tensor(dims).type_as(target_boxes)
                scale_fct = scale_fct.unsqueeze(0).repeat(len(target_boxes), 1)
                target_boxes = target_boxes * scale_fct

            # Plots prediction and ground-truth boxes and saves image.
            save_infer_img(
                img,
                self.hparams.imgs_dir,
                name,
                self.class_names,
                boxes,
                confs,
                preds,
                self.hparams.offset,
                target_boxes=target_boxes,
                target_labels=target_labels,
            )

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Performs inference without targets and saves images."""

        imgs = batch[0]
        names = batch[1]
        orig_sizes = batch[2]
        orig_sizes = [torch.tensor(x) for x in orig_sizes]
        orig_sizes = torch.stack(orig_sizes).type_as(imgs.tensors)

        return self.predict_helper(imgs, names, orig_sizes)
     
    @staticmethod
    def add_model_specific_args(parent_parser):
        """Adds model configuration arguments to parser."""

        return add_model_args(parent_parser)

