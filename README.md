# Face recognition framework based on PyTorch.


### Introduction

This is a face recognition framework based on PyTorch with convenient training, evaluation and feature extraction functions. It is originally a multi-task face recognition framework for our accpeted ECCV 2018 paper, "Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition". However, it is also a common framework for face recognition. You can freely customize your experiments with your data and configurations with it.

### Paper

Xiaohang Zhan, Ziwei Liu, Junjie Yan, Dahua Lin, Chen Change Loy, ["Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition"](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaohang_Zhan_Consensus-Driven_Propagation_in_ECCV_2018_paper.pdf), ECCV 2018

Project Page:
[link](http://mmlab.ie.cuhk.edu.hk/projects/CDP/)

### Why multi-task?

Different datasets have different identity (category) sets. We do not know the intersection between them. Hence instead of merging identity sets of different datasets, regarding them as different tasks is an effective alternative way.

### Features

Framework: Multi-task, Single Task

Loss: Softmax Loss, ArcFace

Backbone CNN: ResNet, DenseNet, Inception, InceptionResNet, NASNet, VGG

Benchmarks: Megaface (FaceScrub), IJB-A, LFW

Data aug: flip, scale, translation

Online testing and visualization with Tensorboard.

### Setup step by step

1. Clone the project.

      ```
      git clone git@github.com:XiaohangZhan/face_recognition_framework.git
      cd face_recognition_framework
      ```

2. Dependency.

    python>=3.6, tensorboardX, pytorch>=0.3.1

3. Data Preparation.

      Download datasets from [insightface](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) into your data storage folder, e.g., `~/data/face_recognition/`. Taking CASIA-Webface for example:

      ```sh
      cd ~/data/face_recognition/
      unzip faces_CASIA_112x112.zip
      cd - # back to the repo root
      python tools/convert_data.py -r ~/data/face_recognition/faces_webface_112x112 -o ~/data/face_recognition/faces_webface_112x112 # convert mxnet records into images
      ln -s ~/data/face_recognition/faces_webface_112x112 data/webface
      ```

4. Training.

      ```
      sh experiments/webface/res50arc-bs64-sz224-ep35/train.sh
      ```

5. Evalution.

      ```
      sh experiments/webface/res50arc-bs64-sz224-ep35/evaluation.sh
      ```

6. Feature extraction.

      Firstly, specify the `data_root` and `data_list` under `extract_info` in the config file. The `data_list` is a txt file containing image relative filename in each line. Then execute:

      ```
      sh experiments/webface/res50arc-bs64-sz224-ep35/extract.sh # feature file is stored in checkpoints/ckpt_epoch_*_*.bin
      ```

### Baselines

* Trained using Webface, resnet50

| arch      | LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) |
| --------- | ------ | --------- | --------- | ----------- | -------- | -------- |
| resnet-50 | 0.9850 | 0.9804    | 0.9117    | 0.8967      | 0.9013   | 0.8423   |

### Bibtex

If you find this code useful in your research, please cite:
```
@inproceedings{zhan2018consensus,
  title={Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition},
  author={Zhan, Xiaohang and Liu, Ziwei and Yan, Junjie and Lin, Dahua and Change Loy, Chen},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={568--583},
  year={2018}
}
```

### TODO
