#!/usr/bin/env python3
"""Mask2Former TensorRT demo without torch2trt.

This script exports the inference core to ONNX, maps Mask2Former's
MSDeformAttn autograd Function to PoseMultiscaleDeformableAttnPlugin_TRT,
builds a TensorRT engine with the modern tensor-address API, and runs image
inference with Detectron2 visualization.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import os
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
FUSION_ROOT = Path(__file__).resolve().parent
PLUGIN_ROOT = FUSION_ROOT / "trt_plugins" / "ms_deform_attn"
TRT_DIR = Path(os.environ.get("TENSORRT_DIR", "/root/cys/DRIVER/TensorRT-11.0.0.114"))
PLUGIN_OP_TYPE = "PoseMultiscaleDeformableAttnPlugin_TRT"
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "pose_custom"
INPUT_NAME = "images"
LOGITS_NAME = "pred_logits"
MASKS_NAME = "pred_masks"
WINDOW_NAME = "mask2former trt demo"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "demo"))
sys.path.insert(0, str(PLUGIN_ROOT))

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

from mask2former import add_maskformer2_config
from mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from mask2former.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import MSDeformAttnFunction
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoder,
)
from predictor import VisualizationDemo
from load_plugin_lib import load_plugin_lib


def _preload_tensorrt_libs() -> None:
    lib_dir = TRT_DIR / "lib"
    if not lib_dir.is_dir():
        return
    for lib_name in ("libnvinfer.so", "libnvinfer_plugin.so", "libnvonnxparser.so"):
        candidates = sorted(lib_dir.glob(f"{lib_name}*"))
        if candidates:
            ctypes.CDLL(str(candidates[-1]), mode=ctypes.RTLD_GLOBAL)


_PLUGIN_HANDLE: ctypes.CDLL | None = None


def load_msda_plugin(enable_optimized_kernel: bool | None = None) -> str:
    global _PLUGIN_HANDLE
    plugin_path = load_plugin_lib()
    if _PLUGIN_HANDLE is None:
        try:
            _PLUGIN_HANDLE = ctypes.CDLL(plugin_path, winmode=0)
        except TypeError:
            _PLUGIN_HANDLE = ctypes.CDLL(plugin_path)

    if enable_optimized_kernel is not None:
        try:
            setter = _PLUGIN_HANDLE.pose_msda_set_enable_optimized_kernel
            getter = _PLUGIN_HANDLE.pose_msda_get_enable_optimized_kernel
        except AttributeError as exc:
            raise RuntimeError(
                "插件库缺少 pose_msda_set/get_enable_optimized_kernel 符号，请先重新编译 "
                "fusion_engine/trt_plugins/ms_deform_attn"
            ) from exc
        setter.argtypes = [ctypes.c_int]
        setter.restype = None
        getter.argtypes = []
        getter.restype = ctypes.c_int
        expected = 1 if enable_optimized_kernel else 0
        setter(expected)
        actual = getter()
        if actual != expected:
            raise RuntimeError(f"MSDA optimized kernel switch failed: expected={expected}, actual={actual}")
    return plugin_path


def setup_cfg(args: argparse.Namespace):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Mask2Former TensorRT demo without torch2trt")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", required=True, help="input image path(s), or one glob pattern")
    parser.add_argument(
        "--output",
        default=str(FUSION_ROOT / "output"),
        help="output file or directory for visualized results",
    )
    parser.add_argument("--onnx", default=None, help="ONNX path; default depends on first preprocessed image shape")
    parser.add_argument("--engine", default=None, help="TensorRT engine path; default depends on first preprocessed image shape")
    parser.add_argument("--force-export", action="store_true", help="always export ONNX even if it already exists")
    parser.add_argument("--force-build", action="store_true", help="always rebuild TensorRT engine even if it already exists")
    parser.add_argument("--fp16", action="store_true", help="shortcut for --onnx-precision fp16 and TensorRT FP16 build flag")
    parser.add_argument(
        "--onnx-precision",
        choices=("fp32", "fp16"),
        default=None,
        help="ONNX export precision. Defaults to fp16 when --fp16 is set, otherwise fp32",
    )
    parser.add_argument("--workspace-gb", type=float, default=4.0, help="TensorRT workspace size in GiB")
    parser.add_argument("--input-size", type=int, default=800, help="fixed square TensorRT input size, e.g. 800 means 800x800")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--warm-iters", type=int, default=10, help="warmup iterations for benchmark")
    parser.add_argument("--bench-iters", type=int, default=50, help="benchmark iterations; <=0 disables benchmark")
    parser.add_argument("--no-optimized-msda", action="store_true", help="disable plugin optimized kernel switch")
    parser.add_argument("--show", action="store_true", help="show result with OpenCV when no output is provided")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="kept for demo.py CLI compatibility")
    parser.add_argument(
        "--opts",
        help="Modify config options using command-line KEY VALUE pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class ExportableMask2Former(torch.nn.Module):
    """Tensor-only inference wrapper: normalized/padded image -> logits and masks."""

    def __init__(self, model: torch.nn.Module, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.sem_seg_head = model.sem_seg_head
        self.output_size = output_size

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        outputs = self.sem_seg_head(features)
        pred_logits = outputs["pred_logits"]
        pred_masks = F.interpolate(
            outputs["pred_masks"],
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )
        return pred_logits, pred_masks


def register_msda_symbolic() -> None:
    def symbolic(g, value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        int32_type = 6  # onnx.TensorProto.INT32
        spatial_shapes_i32 = g.op("Cast", spatial_shapes, to_i=int32_type)
        level_start_index_i32 = g.op("Cast", level_start_index, to_i=int32_type)
        plugin_out = g.op(
            f"nvidia::{PLUGIN_OP_TYPE}",
            value,
            spatial_shapes_i32,
            level_start_index_i32,
            sampling_locations,
            attention_weights,
            plugin_version_s=PLUGIN_VERSION,
            plugin_namespace_s=PLUGIN_NAMESPACE,
        )
        # TensorRT plugin outputs [B, Q, num_heads, channels_per_head]. The PyTorch
        # module expects [B, Q, hidden_dim] before output_proj.
        shape = g.op("Constant", value_t=torch.tensor([0, 0, -1], dtype=torch.long))
        return g.op("Reshape", plugin_out, shape)

    MSDeformAttnFunction.symbolic = staticmethod(symbolic)


def _trt_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
    decoder_output = self.decoder_norm(output).transpose(0, 1)
    outputs_class = self.class_embed(decoder_output)
    mask_embed = self.mask_embed(decoder_output)
    outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

    attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
    repeated_mask = attn_mask.sigmoid().flatten(2).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
    bool_mask = repeated_mask < 0.5
    all_true = bool_mask.sum(-1, keepdim=True) == bool_mask.shape[-1]
    bool_mask = torch.where(all_true, torch.zeros_like(bool_mask), bool_mask)
    float_mask = torch.where(
        bool_mask,
        torch.full_like(repeated_mask, float("-inf")),
        torch.zeros_like(repeated_mask),
    )
    return outputs_class, outputs_mask, float_mask.detach().flatten(0, 1)


def _trt_decoder_forward(self, x, mask_features, mask=None):
    assert len(x) == self.num_feature_levels
    src = []
    pos = []
    size_list = []
    del mask

    for i in range(self.num_feature_levels):
        size_list.append(x[i].shape[-2:])
        pos.append(self.pe_layer(x[i], None).flatten(2))
        src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
        pos[-1] = pos[-1].permute(2, 0, 1)
        src[-1] = src[-1].permute(2, 0, 1)

    _, batch_size, _ = src[0].shape
    query_embed = self.query_embed.weight.unsqueeze(1).expand(-1, batch_size, -1)
    output = self.query_feat.weight.unsqueeze(1).expand(-1, batch_size, -1)

    outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
        output, mask_features, attn_mask_target_size=size_list[0]
    )

    for i in range(self.num_layers):
        level_index = i % self.num_feature_levels
        output = self.transformer_cross_attention_layers[i](
            output,
            src[level_index],
            memory_mask=attn_mask,
            memory_key_padding_mask=None,
            pos=pos[level_index],
            query_pos=query_embed,
        )
        output = self.transformer_self_attention_layers[i](
            output,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=query_embed,
        )
        output = self.transformer_ffn_layers[i](output)
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
        )

    return {"pred_logits": outputs_class, "pred_masks": outputs_mask}


def _trt_pixel_decoder_forward_features(self, features):
    srcs = []
    pos = []
    for idx, f in enumerate(self.transformer_in_features[::-1]):
        x = features[f]
        srcs.append(self.input_proj[idx](x))
        pos.append(self.pe_layer(x))

    y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
    batch_size = y.shape[0]

    split_size_or_sections = [None] * self.transformer_num_feature_levels
    for i in range(self.transformer_num_feature_levels):
        if i < self.transformer_num_feature_levels - 1:
            split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
        else:
            split_size_or_sections[i] = y.shape[1] - level_start_index[i]
    y = torch.split(y, split_size_or_sections, dim=1)

    out = []
    multi_scale_features = []
    num_cur_levels = 0
    for i, z in enumerate(y):
        out.append(z.transpose(1, 2).view(batch_size, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

    for idx, f in enumerate(self.in_features[: self.num_fpn_levels][::-1]):
        x = features[f]
        cur_fpn = self.lateral_convs[idx](x)
        y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
        y = self.output_convs[idx](y)
        out.append(y)

    for output in out:
        if num_cur_levels < self.maskformer_num_feature_levels:
            multi_scale_features.append(output)
            num_cur_levels += 1

    return self.mask_features(out[-1]), out[0], multi_scale_features


def patch_decoder_for_export(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, MultiScaleMaskedTransformerDecoder):
            module.forward = MethodType(_trt_decoder_forward, module)
            module.forward_prediction_heads = MethodType(_trt_forward_prediction_heads, module)
        elif isinstance(module, MSDeformAttnPixelDecoder):
            module.forward_features = MethodType(_trt_pixel_decoder_forward_features, module)


def preprocess_image(
    original_bgr: np.ndarray,
    cfg,
    model: torch.nn.Module,
    input_size: int,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    image = original_bgr
    if cfg.INPUT.FORMAT == "RGB":
        image = image[:, :, ::-1]

    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1), device=model.device)
    tensor = (tensor - model.pixel_mean) / model.pixel_std
    image_size = (input_size, input_size)
    original_size = (original_bgr.shape[0], original_bgr.shape[1])
    return tensor.unsqueeze(0).contiguous(), image_size, original_size


def draw_predictions(predictions: dict[str, Any], image_bgr: np.ndarray, cfg) -> Any:
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
    visualizer = Visualizer(image_bgr[:, :, ::-1], metadata, instance_mode=ColorMode.IMAGE)
    if "panoptic_seg" in predictions:
        panoptic_seg, segments_info = predictions["panoptic_seg"]
        return visualizer.draw_panoptic_seg_predictions(panoptic_seg.cpu(), segments_info)
    if "sem_seg" in predictions:
        return visualizer.draw_sem_seg(predictions["sem_seg"].argmax(dim=0).cpu())
    if "instances" in predictions:
        return visualizer.draw_instance_predictions(predictions=predictions["instances"].to("cpu"))
    raise RuntimeError("prediction dict has no supported visualization key")


def postprocess_outputs(
    model: torch.nn.Module,
    cfg,
    pred_logits: torch.Tensor,
    pred_masks: torch.Tensor,
    image_size: tuple[int, int],
    original_size: tuple[int, int],
) -> dict[str, Any]:
    height, width = original_size
    mask_cls = pred_logits[0]
    mask_pred = pred_masks[0]
    mask_pred = sem_seg_postprocess(mask_pred, image_size, height, width)

    result: dict[str, Any] = {}
    if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
        result["sem_seg"] = model.semantic_inference(mask_cls, mask_pred)
    if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
        result["panoptic_seg"] = model.panoptic_inference(mask_cls, mask_pred)
    if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
        result["instances"] = model.instance_inference(mask_cls, mask_pred)
    return result


def export_onnx(
    model: torch.nn.Module,
    onnx_path: Path,
    dummy_input: torch.Tensor,
    opset: int,
    force: bool,
    precision: str,
) -> None:
    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"unsupported ONNX export precision: {precision}")

    if onnx_path.is_file() and not force:
        print(f"[ONNX] reuse {onnx_path} ({precision})")
        return

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    register_msda_symbolic()
    patch_decoder_for_export(model)

    export_dtype = torch.float16 if precision == "fp16" else torch.float32
    dummy_input = dummy_input.to(dtype=export_dtype).contiguous()
    export_model = ExportableMask2Former(model, output_size=tuple(dummy_input.shape[-2:]))
    export_model = export_model.to(device=dummy_input.device, dtype=export_dtype).eval()

    print(f"[ONNX] export {onnx_path} precision={precision}")
    kwargs = dict(
        input_names=[INPUT_NAME],
        output_names=[LOGITS_NAME, MASKS_NAME],
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        custom_opsets={"nvidia": opset},
    )
    with torch.inference_mode():
        try:
            torch.onnx.export(export_model, (dummy_input,), str(onnx_path), dynamo=False, **kwargs)
        except TypeError:
            kwargs.pop("dynamo", None)
            torch.onnx.export(export_model, (dummy_input,), str(onnx_path), **kwargs)
    print(f"[ONNX] exported ({precision}): {onnx_path}")


def _get_plugin_creator(trt, logger):
    trt.init_libnvinfer_plugins(logger, "")
    if hasattr(trt, "get_builder_plugin_registry") and hasattr(trt, "EngineCapability"):
        registry = trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD)
    else:
        registry = trt.get_plugin_registry()
    creator = registry.get_creator(PLUGIN_OP_TYPE, PLUGIN_VERSION, PLUGIN_NAMESPACE)
    if creator is None:
        raise RuntimeError(
            f"TensorRT plugin creator not registered: {PLUGIN_OP_TYPE}, "
            f"version={PLUGIN_VERSION}, namespace={PLUGIN_NAMESPACE}"
        )
    return creator


def build_or_load_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_gb: float,
    force_build: bool,
):
    _preload_tensorrt_libs()
    import tensorrt as trt

    load_msda_plugin()
    logger = trt.Logger(trt.Logger.INFO)
    _get_plugin_creator(trt, logger)

    if engine_path.is_file() and not force_build:
        print(f"[TRT] load {engine_path}")
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        return engine

    print(f"[TRT] build {engine_path}")
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    else:
        config.max_workspace_size = int(workspace_gb * (1 << 30))

    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 3

    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errors}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("TensorRT engine deserialize failed after build")
    return engine


def trt_dtype_to_torch(dtype) -> torch.dtype:
    import tensorrt as trt

    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.INT32:
        return torch.int32
    if dtype == trt.DataType.INT8:
        return torch.int8
    if hasattr(trt.DataType, "BOOL") and dtype == trt.DataType.BOOL:
        return torch.bool
    raise TypeError(f"unsupported TensorRT dtype: {dtype}")


class TrtRunner:
    def __init__(self, engine) -> None:
        import tensorrt as trt

        self.trt = trt
        self.engine = engine
        self.context = engine.create_execution_context()
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        if INPUT_NAME not in self.input_names:
            raise RuntimeError(f"TensorRT engine inputs={self.input_names}, expected input '{INPUT_NAME}'")

    def infer(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if not images.is_cuda:
            images = images.cuda()
        expected_dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(INPUT_NAME))
        if images.dtype != expected_dtype:
            images = images.to(expected_dtype)
        images = images.contiguous()

        self.context.set_input_shape(INPUT_NAME, tuple(images.shape))
        self.context.set_tensor_address(INPUT_NAME, images.data_ptr())
        missing = self.context.infer_shapes()
        if missing:
            raise RuntimeError(f"TensorRT shape inference has missing tensors: {missing}")

        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            tensor = torch.empty(shape, dtype=dtype, device=images.device)
            self.context.set_tensor_address(name, tensor.data_ptr())
            outputs[name] = tensor

        ok = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        return outputs


def resolve_inputs(patterns: list[str]) -> list[str]:
    if len(patterns) == 1:
        expanded = glob.glob(os.path.expanduser(patterns[0]))
        if expanded:
            return sorted(expanded)
    return [os.path.expanduser(path) for path in patterns]


def resolve_onnx_precision(args: argparse.Namespace) -> str:
    if args.onnx_precision is not None:
        return args.onnx_precision
    return "fp16" if args.fp16 else "fp32"


def default_artifact_paths(input_tensor: torch.Tensor, args: argparse.Namespace) -> tuple[Path, Path]:
    height, width = input_tensor.shape[-2:]
    build_dir = FUSION_ROOT / "build"
    precision = resolve_onnx_precision(args)
    stem = f"mask2former_{height}x{width}_{precision}"
    onnx_path = Path(args.onnx) if args.onnx else build_dir / f"{stem}.onnx"
    engine_path = Path(args.engine) if args.engine else build_dir / f"{stem}.engine"
    return onnx_path, engine_path


def save_or_show(vis_output, input_path: str, args: argparse.Namespace) -> None:
    if args.output:
        output_path = Path(args.output)
        if len(resolve_inputs(args.input)) > 1 or output_path.suffix == "":
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / Path(input_path).name
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        vis_output.save(str(output_path))
        print(f"[SAVE] {output_path}")
        return

    if args.show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, vis_output.get_image()[:, :, ::-1])
        cv2.waitKey(0)


def benchmark(runner: TrtRunner, input_tensor: torch.Tensor, warm_iters: int, bench_iters: int) -> None:
    if bench_iters <= 0:
        return
    for _ in range(warm_iters):
        runner.infer(input_tensor)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(bench_iters):
        runner.infer(input_tensor)
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / bench_iters
    print(f"[BENCH] TensorRT latency={avg_ms:.3f} ms FPS={1000.0 / avg_ms:.2f}")


def main() -> int:
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    _preload_tensorrt_libs()
    load_msda_plugin(enable_optimized_kernel=not args.no_optimized_msda)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT inference")

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    model = demo.predictor.model.eval().cuda()

    input_paths = resolve_inputs(args.input)
    if not input_paths:
        raise FileNotFoundError(f"input path(s) not found: {args.input}")

    runners: dict[tuple[int, ...], TrtRunner] = {}
    benchmarked_shapes: set[tuple[int, ...]] = set()

    for path in tqdm.tqdm(input_paths):
        image = read_image(path, format="BGR")
        input_tensor, image_size, original_size = preprocess_image(image, cfg, model, args.input_size)
        shape_key = tuple(input_tensor.shape)

        if shape_key not in runners:
            precision = resolve_onnx_precision(args)
            onnx_path, engine_path = default_artifact_paths(input_tensor, args)
            export_onnx(model, onnx_path, input_tensor, args.opset, args.force_export, precision)
            engine = build_or_load_engine(
                onnx_path, engine_path, args.workspace_gb, args.force_build
            )
            runners[shape_key] = TrtRunner(engine)
            print(f"[TRT] prepared runner for input shape {shape_key}")

        runner = runners[shape_key]
        if shape_key not in benchmarked_shapes:
            benchmark(runner, input_tensor, args.warm_iters, args.bench_iters)
            benchmarked_shapes.add(shape_key)

        start_time = time.time()
        outputs = runner.infer(input_tensor)
        torch.cuda.synchronize()
        pred_logits = outputs[LOGITS_NAME].float()
        pred_masks = outputs[MASKS_NAME].float()
        predictions = postprocess_outputs(model, cfg, pred_logits, pred_masks, image_size, original_size)
        vis_output = draw_predictions(predictions, image, cfg)
        logger.info(f"{path}: finished TensorRT inference in {time.time() - start_time:.3f}s")
        save_or_show(vis_output, path, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
