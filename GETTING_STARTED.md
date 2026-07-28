## Mask2Former 入门指南

本文档简要介绍了 Mask2Former 的用法。

请看 [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) 充分利用。


### 使用预训练模型进行推理演示

1. 从中选择一个模型及其配置文件
  [model zoo](MODEL_ZOO.md)，
  例如， `configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml`。
2. 我们提供 `demo.py` 它可以演示内置配置。运行命令如下：
```
cd demo/
python demo.py --config-file ../configs/coco/panoptic-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml \
  --input ../model/1.jpg ../model/2.jpg \
  --output ../model/output_pan \
  --opts MODEL.WEIGHTS ../model/model_final_9fd0ae.pkl

python demo.py --config-file ../configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml \
  --input ../model/1.jpg ../model/2.jpg \
  --output ../model/output_ins \
  --opts MODEL.WEIGHTS ../model/model_final_86143f.pkl
```
这些配置是为训练而设计的，因此我们需要指定 `MODEL.WEIGHTS` 从模型库中选择模型进行评估。
该命令将运行推理并在 OpenCV 窗口中显示可视化结果。

有关命令行参数的详细信息，请参阅 `demo.py -h` 或者查看其源代码
为了理解其行为，一些常见的论点包括：
* 要在您的网络摄像头上运行__，请替换 `--input files` 和 `--webcam`。
* 要在视频上运行__，请替换 `--input files` 和 `--video-input video.mp4`。
* 要在 CPU 上运行，请添加 `MODEL.DEVICE cpu` 后 `--opts`。
* 要将输出保存到目录（图像）或文件（网络摄像头或视频），请使用 `--output`。


### 命令行训练与评估

我们提供一份脚本 `train_net.py`，其目的是训练 Mask2Former 中提供的所有配置。

要使用“train_net.py”训练模型，首先
按以下步骤设置相应的数据集
[datasets/README.md](./datasets/README.md)，
然后运行：
```
python train_net.py --num-gpus 8 \
  --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml
```

这些配置是为 8 GPU 训练而设计的。
由于我们使用 ADAMW 优化器，因此不清楚如何根据批次大小调整学习率。
要在单个 GPU 上进行训练，您需要自行确定学习率和批次大小：
```
python train_net.py \
  --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH SET_TO_SOME_REASONABLE_VALUE SOLVER.BASE_LR SET_TO_SOME_REASONABLE_VALUE
```

要评估模型的性能，请使用
```
python train_net.py \
  --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
更多选项，请参见 `python train_net.py -h`。


### 视频实例分割
请使用 `demo_video/demo.py` 用于视频实例分割演示和 `train_net_video.py` 训练
并评估视频实例分割模型。
