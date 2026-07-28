## 安装

### 要求

- Linux 或 macOS，Python 版本 ≥ 3.6
- PyTorch ≥ 1.9 和 [torchvision](https://github.com/pytorch/vision/) 这与 PyTorch 的安装方式相符。
  它们一起安装 [pytorch.org](https://pytorch.org) 为了确保这一点。注意，请检查
  PyTorch 版本与 Detectron2 所需的版本匹配。
- 检测器2：跟随 [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html)。
- OpenCV 是可选的，但演示和可视化需要它。
- `pip install -r requirements.txt`

### MSDeformAttn 的 CUDA 内核
准备好所需环境后，运行以下命令为 MSDeformAttn 编译 CUDA 内核：

`CUDA_HOME` 必须进行定义，并指向已安装的 CUDA 工具包的目录。

```bash
export CUDA_HOME=/usr/local/cuda
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

#### 在另一个系统的基础上构建
在没有GPU设备的系统上构建，但需要提供驱动程序：
```bash
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
```

### conda 环境设置示例
```bash
conda create --name mask2former python=3.8 -y
conda activate mask2former
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
git clone git@github.com:facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```
