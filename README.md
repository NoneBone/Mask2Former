# Mask2Former：用于通用图像分割的带注意力掩码转换器（CVPR 2022）

[Bowen Cheng](https://bowenc0221.github.io/)， [Ishan Misra](https://imisra.github.io/)， [Alexander G. Schwing](https://alexander-schwing.de/)， [Alexander Kirillov](https://alexander-kirillov.github.io/)， [Rohit Girdhar](https://rohitgirdhar.github.io/)

[[`arXiv`](https://arxiv.org/abs/2112.01527)] [[`Project`](https://bowenc0221.github.io/mask2former)] [[`BibTeX`](#CitingMask2Former)]

<div align="center">
  <img src="https://bowenc0221.github.io/images/maskformerv2_teaser.png" width="100%" height="100%"/>
</div><br/>

### 特征
* 一种用于全景分割、实例分割和语义分割的单一架构。
* 支持主要分割数据集：ADE20K、Cityscapes、COCO、Mapillary Vistas。

## 更新
* 添加 Google Colab 演示。
* 现在支持视频实例分割！请查看我们的 [tech report](https://arxiv.org/abs/2112.10764) 更多详情请见下文。

## 安装

看 [installation instructions](INSTALL.md)。

## 入门

看 [Preparing Datasets for Mask2Former](datasets/README.md)。

看 [Getting Started with Mask2Former](GETTING_STARTED.md)。

使用 Colab 运行我们的演示： [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uIWE5KbGFSjrxey2aRd5pWkKNY1_SaNq)

整合到 [Huggingface Spaces 🤗](https://huggingface.co/spaces) 使用 [Gradio](https://github.com/gradio-app/gradio)试试网页演示： [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/Mask2Former)

复现的网页演示和 Docker 镜像可在此处获取： [![Replicate](https://replicate.com/facebookresearch/mask2former/badge)](https://replicate.com/facebookresearch/mask2former)

## 高级用法

看 [Advanced Usage of Mask2Former](ADVANCED_USAGE.md)。

## 模型库和基线

我们提供大量基准结果和训练好的模型，可供下载。 [Mask2Former Model Zoo](MODEL_ZOO.md)。

## 执照

盾： [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Mask2Former 的大部分内容均已获得许可。 [MIT License](LICENSE)。


然而，该项目的部分内容以单独的许可条款提供：Swin-Transformer-Semantic-Segmentation 的许可条款如下： [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE)Deformable-DETR 已获得以下许可： [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE)。

## <a name="CitingMask2Former"></a>引用 Mask2Former

如果您在研究中使用 Mask2Former，或者希望参考已发表的基线结果， [Model Zoo](MODEL_ZOO.md)请使用以下 BibTeX 条目。

```BibTeX
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
```

如果您觉得这段代码有用，也请参考以下 BibTeX 条目。

```BibTeX
@inproceedings{cheng2021maskformer,
  title={Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author={Bowen Cheng and Alexander G. Schwing and Alexander Kirillov},
  journal={NeurIPS},
  year={2021}
}
```

## 致谢

代码主要基于 MaskFormer（https://github.com/facebookresearch/MaskFormer）。
