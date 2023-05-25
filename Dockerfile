FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

RUN apt-get update

RUN apt-get -y install libgl1-mesa-glx libsm6 libxext6 ffmpeg gcc sudo wget vim git tmux unzip ninja-build

RUN pip install azureml-core==1.35.0post1
RUN pip install configargparse==1.5.3
RUN pip install gdown==4.4.0
RUN pip install numpy==1.21.0
RUN pip install opencv-contrib-python==4.5.4.60
RUN pip install pycocotools==2.0.2
RUN pip install pytorch-lightning==1.5.1
RUN pip install scipy==1.7.1
RUN pip install sparsemax==0.1.9
RUN pip install tensorboard==2.7.0

RUN conda uninstall torchvision
RUN conda install torchvision==0.10.0 -c pytorch
