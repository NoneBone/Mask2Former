## Mask2Former Trt Demo

## demo_trt
```
python fusion_engine/demo_trt.py \
--config-file configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml \
--input model/1.jpg model/2.jpg \
--output model/output_trt \
--fp16 \
--force-export \
--force-build \
--opts MODEL.WEIGHTS model/model_final_86143f.pkl
```

## msda 插件载入

```sh
make -C trt_plugins/ms_deform_attn clean

make -C trt_plugins/ms_deform_attn \
  BUILD=release \
  TENSORRT_DIR=/root/cys/DRIVER/TensorRT-11.0.0.114 \
  CUDA_INSTALL_DIR=/usr/local/cuda

编译完成后产物 libpose_ms_deform_attn_plugin.so 位于：

ls ./trt_plugins/ms_deform_attn/

可选验证命令：

LD_LIBRARY_PATH=/root/cys/DRIVER/TensorRT-11.0.0.114/lib:$LD_LIBRARY_PATH \
/opt/conda/envs/cp312/bin/python -c "
import ctypes, tensorrt as trt
ctypes.CDLL('trt_plugins/ms_deform_attn/libpose_ms_deform_attn_plugin.so')
trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.ERROR), '')
reg = trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD)
creator = reg.get_creator('PoseMultiscaleDeformableAttnPlugin_TRT', '1', 'pose_custom')
print('creator_found', creator is not None)
print(None if creator is None else (creator.name, creator.plugin_version, creator.plugin_namespace))"
```

