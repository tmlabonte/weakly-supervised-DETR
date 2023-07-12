# Weakly Supervised Detection Transformer (WS-DETR)

![WS-DETR Architecture](ws-detr.png)

Official codebase for the paper: [Scaling Novel Object Detection with Weakly Supervised Detection Transformers.](https://arxiv.org/abs/2207.05205)

WS-DETR leverages large-scale fully supervised pretraining to detect hundreds of novel classes using only image-level classification labels.

### Setup Instructions
First, you will need to [install Docker](https://docs.docker.com/engine/install/ubuntu/) if it is not already available on your machine. To download the WS-DETR Docker image and build the deformable attention modules, use:

```
git clone https://github.com/tmlabonte/weakly-supervised-DETR
cd weakly-supervised-DETR
sudo docker run -it --gpus all --privileged --shm-size 32g -v $(pwd):/local tmlabonte/ws-detr:latest
cd /local
sh make.sh
```

To download the FGVC-Aircraft and FSOD datasets, use the following command. You can also download the iNaturalist 2017 dataset, but it is quite large, so we suggest starting with FGVC-Aircraft and FSOD.

`python download.py --datasets fgvc`

To download our FSOD-800 pretrained Deformable DETR checkpoints, use:

`gdown https://drive.google.com/drive/folders/1ZJIElm5A7TaZtIvWjaqnNet2llDQVwGq -O ckpts --folder`

### Quick Start
To train WS-DETR on the FSOD-200 novel classes for 1 epoch on a single 16GB GPU, use the following command. Note that WS-DETR Full (with our joint probability estimation and sparsity techniques) is enabled by default.

`python src/main.py -c cfgs/quickstart.yaml`

The model will automatically visualize the output and save it to `out/imgs`.

### Configs
Config files are located in the `cfgs/` directory. The pre-made configs correspond to experiments from the paper. To run a new experiment, you can make a new config or just use command line arguments:

`python src/main.py -c cfgs/fsod_split0.yaml --batch_size 1`

All PyTorch Lightning 1.5 [Trainer options](https://pytorch-lightning.readthedocs.io/en/1.5.1/common/trainer.html#trainer-flags) are valid config variables. For an explanation of what each config variable does, use `python src/main.py -h`.

### Testing and Inference
To test on a labeled dataset, set the `test` task. To predict on a directory of images, set the `infer` task. For example,

`python src/main.py -c cfgs/quickstart.yaml --task test --weights out/version_0/checkpoints/last.ckpt`

You can also test with a class-agnostic model (e.g., to visualize the boxes before training) as follows:

`python src/main.py -c cfgs/quickstart.yaml --task test --weights ckpts/deformable-detr_fsod-800_class-agnostic_50epochs.pth --supervised --classes 2`

Note that the labeled datasets given have no infer directory.

### Multi-GPU Training
To perform multi-GPU training, simply set the `gpus` argument:

`python src/main.py -c cfgs/fsod_split0.yaml --gpus 8`

Note that `batch_size` is per-GPU. If training on less than 8 GPUs, set the `accumulate_grad_batches` option to increase the effective batch size:

`python src/main.py -c cfgs/fsod_split0.yaml --gpus 2 --accumulate_grad_batches 4`

The effective batch size is `gpus` x `batch_size` x `accumulate_grad_batches`. We use a default batch size of 2 per GPU (for 16GB GPUs) and an effective batch size of 16. We have found that a batch size of 32 also works well.

### Adding Your Dataset
Training WS-DETR on your own dataset is simple. First, you will need a [COCO-style annotation file](https://cocodataset.org/#format-data). If you are labeling your own dataset, you can make a `classes` field for each image which contains a list of category IDs present in the image, or you can make a box annotation for each category ID and our code will convert it for you. Second, make a new config file for your experiment following the examples in the `cfgs/` directory. Remember to set the directories and annotations locations at the top, as well as the `classes` field. If your category IDs are 1-indexed, set `offset: 1`. To use `ReduceLROnPlateau` scheduler instead of `StepLR` scheduler, set `lr_patience`. Finally, to run your experiment, use the command:

`python src/main.py -c cfgs/my_config.yaml`

### Suppressing Caffe2 Warning
When training, especially with multiple GPUs and workers, you may see this warning:

`[W pthreadpool-cpp.cc:88] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)`

This is a harmless warning due to our version of PyTorch. To suppress it, simply append `2>/dev/null` to your command to send `stderr` to `/dev/null`:

`python src/main.py -c cfgs/fsod_split0.yaml 2>/dev/null`

Note that this will also send any legitimate error messages to `/dev/null`. For this reason, we recommend debugging on a single GPU with few workers.

### Note on Nondeterminism
Due to the use of `atomicAdd` CUDA operations in the deformable attention module from Deformable DETR, training is inherently nondeterministic even with all seeds set. So, it is unlikely that one can reproduce exactly the results seen in the paper. However, by training with a large batch size and proper learning rate decay, most of this nondeterminism can be mitigated.

### Previous Works
Original [DETR code](https://github.com/facebookresearch/detr) by Facebook.

Original [Deformable DETR code](https://github.com/fundamentalvision/Deformable-DETR) by SenseTime.

We use the [FSOD](https://arxiv.org/abs/1908.01998), [FGVC-Aircraft](https://arxiv.org/abs/1306.5151), and [iNaturalist 2017](https://arxiv.org/abs/1707.06642) datasets. We also use the [sparsemax](https://arxiv.org/abs/1602.02068) function.

From a WSOD perspective, our work builds most heavily on [Uijlings et al. 2018](https://arxiv.org/abs/1708.06128) and [Zhong et al. 2020](https://arxiv.org/abs/2007.07986). Our MIL classifier is based on [WSDDN](https://arxiv.org/abs/1511.02853). Check them out!
