#!/usr/bin/env python3
"""TensorRT MS Deformable Attention plugin correctness and benchmark tests.

本测试只验证 TensorRT plugin 前向输出，不依赖 PyTorch 自定义 CUDA 扩展。
参考输出使用 model/head/ops/test.py 中同等逻辑的纯 PyTorch grid_sample 实现。

运行示例：
  LD_LIBRARY_PATH=/root/cys/DRIVER/TensorRT-11.0.0.114/lib:$LD_LIBRARY_PATH \
    /opt/conda/envs/cp312/bin/python trt_plugins/ms_deform_attn/test_unit.py

只跑性能测试：
  LD_LIBRARY_PATH=/root/cys/DRIVER/TensorRT-11.0.0.114/lib:$LD_LIBRARY_PATH \
    /opt/conda/envs/cp312/bin/python trt_plugins/ms_deform_attn/test_unit.py --skip-check
"""

from __future__ import annotations

import argparse
import ctypes
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from load_plugin_lib import load_plugin_lib


PLUGIN_OP_TYPE = "PoseMultiscaleDeformableAttnPlugin_TRT"
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "pose_custom"
TRT_DIR = Path(os.environ.get("TENSORRT_DIR", "/root/cys/DRIVER/TensorRT-11.0.0.114"))


@dataclass(frozen=True)
class DtypeCase:
    name: str
    torch_dtype: torch.dtype
    onnx_dtype: int
    rtol: float
    atol: float


@dataclass(frozen=True)
class ForwardCase:
    name: str
    batch: int
    num_query: int
    num_heads: int
    channels: int
    num_points: int
    spatial_shapes: tuple[tuple[int, int], ...]

    @property
    def num_levels(self) -> int:
        return len(self.spatial_shapes)


def _preload_tensorrt_libs() -> None:
    """允许在未显式设置 LD_LIBRARY_PATH 时导入 tensorrt Python 包。"""
    lib_dir = TRT_DIR / "lib"
    if not lib_dir.is_dir():
        return

    for lib_name in ("libnvinfer.so", "libnvinfer_plugin.so", "libnvonnxparser.so"):
        candidates = sorted(lib_dir.glob(f"{lib_name}*"))
        if not candidates:
            continue
        ctypes.CDLL(str(candidates[-1]), mode=ctypes.RTLD_GLOBAL)


_PLUGIN_LIB_HANDLE: ctypes.CDLL | None = None


def _load_plugin_handle() -> ctypes.CDLL:
    global _PLUGIN_LIB_HANDLE
    if _PLUGIN_LIB_HANDLE is None:
        plugin_path = load_plugin_lib()
        try:
            _PLUGIN_LIB_HANDLE = ctypes.CDLL(plugin_path, winmode=0)
        except TypeError:
            _PLUGIN_LIB_HANDLE = ctypes.CDLL(plugin_path)

        try:
            _PLUGIN_LIB_HANDLE.pose_msda_set_enable_optimized_kernel.argtypes = [ctypes.c_int]
            _PLUGIN_LIB_HANDLE.pose_msda_set_enable_optimized_kernel.restype = None
            _PLUGIN_LIB_HANDLE.pose_msda_get_enable_optimized_kernel.argtypes = []
            _PLUGIN_LIB_HANDLE.pose_msda_get_enable_optimized_kernel.restype = ctypes.c_int
        except AttributeError as exc:
            raise RuntimeError(
                "当前插件库没有运行时优化开关符号，请先重新编译：\n"
                "  make -C trt_plugins/ms_deform_attn clean\n"
                "  make -C trt_plugins/ms_deform_attn BUILD=release "
                "TENSORRT_DIR=/root/cys/DRIVER/TensorRT-11.0.0.114 CUDA_INSTALL_DIR=/usr/local/cuda"
            ) from exc
    return _PLUGIN_LIB_HANDLE


def set_optimized_kernel_enabled(enable: bool) -> None:
    lib = _load_plugin_handle()
    expected = 1 if enable else 0
    lib.pose_msda_set_enable_optimized_kernel(expected)
    actual = lib.pose_msda_get_enable_optimized_kernel()
    if actual != expected:
        raise RuntimeError(f"failed to set optimized kernel switch: expected={expected}, actual={actual}")


def _kernel_mode_name(enable: bool) -> str:
    return "optimized" if enable else "original"


def ms_deform_attn_core_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """纯 PyTorch 参考实现，来自 model/head/ops/functions/ms_deform_attn_func.py。"""
    batch, _, num_heads, channels = value.shape
    _, num_query, _, num_levels, num_points, _ = sampling_locations.shape
    split_sizes = [int(h.item() * w.item()) for h, w in value_spatial_shapes]
    value_list = value.split(split_sizes, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for level_id, (height, width) in enumerate(value_spatial_shapes):
        height = int(height.item())
        width = int(width.item())
        value_l = (
            value_list[level_id]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch * num_heads, channels, height, width)
        )
        sampling_grid_l = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        sampling_value_list.append(
            F.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )

    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch * num_heads, 1, num_query, num_levels * num_points
    )
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1)
    return output.view(batch, num_heads, channels, num_query).permute(0, 3, 1, 2).contiguous()


def _make_level_start_index(spatial_shapes: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]),
        dim=0,
    ).contiguous()


def _normalize_attention_weights(attention_weights: torch.Tensor) -> torch.Tensor:
    denom = attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    return (attention_weights / denom).contiguous()


def make_inputs(
    forward_case: ForwardCase,
    dtype: torch.dtype,
    seed: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    spatial_shapes = torch.as_tensor(forward_case.spatial_shapes, dtype=torch.int32, device="cuda").contiguous()
    level_start_index = _make_level_start_index(spatial_shapes)
    spatial_size = int(spatial_shapes.prod(1).sum().item())

    value = (
        torch.rand(
            forward_case.batch,
            spatial_size,
            forward_case.num_heads,
            forward_case.channels,
            device="cuda",
            dtype=dtype,
        )
        * 0.01
    ).contiguous()
    sampling_locations = torch.rand(
        forward_case.batch,
        forward_case.num_query,
        forward_case.num_heads,
        forward_case.num_levels,
        forward_case.num_points,
        2,
        device="cuda",
        dtype=dtype,
    ).contiguous()
    attention_weights = _normalize_attention_weights(
        torch.rand(
            forward_case.batch,
            forward_case.num_query,
            forward_case.num_heads,
            forward_case.num_levels,
            forward_case.num_points,
            device="cuda",
            dtype=dtype,
        )
        + 1e-5
    )
    return value, spatial_shapes, level_start_index, sampling_locations, attention_weights


def build_onnx(
    onnx_path: Path,
    dtype_case: DtypeCase,
    forward_case: ForwardCase,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
) -> None:
    import onnx
    from onnx import helper, numpy_helper

    spatial_size = int(spatial_shapes.prod(1).sum().item())

    value_info = helper.make_tensor_value_info(
        "value",
        dtype_case.onnx_dtype,
        [forward_case.batch, spatial_size, forward_case.num_heads, forward_case.channels],
    )
    sampling_info = helper.make_tensor_value_info(
        "sampling_locations",
        dtype_case.onnx_dtype,
        [
            forward_case.batch,
            forward_case.num_query,
            forward_case.num_heads,
            forward_case.num_levels,
            forward_case.num_points,
            2,
        ],
    )
    weight_info = helper.make_tensor_value_info(
        "attention_weights",
        dtype_case.onnx_dtype,
        [
            forward_case.batch,
            forward_case.num_query,
            forward_case.num_heads,
            forward_case.num_levels,
            forward_case.num_points,
        ],
    )
    output_info = helper.make_tensor_value_info(
        "output",
        dtype_case.onnx_dtype,
        [forward_case.batch, forward_case.num_query, forward_case.num_heads, forward_case.channels],
    )

    node = helper.make_node(
        PLUGIN_OP_TYPE,
        inputs=[
            "value",
            "spatial_shapes",
            "level_start_index",
            "sampling_locations",
            "attention_weights",
        ],
        outputs=["output"],
        domain="nvidia",
        name=f"{PLUGIN_OP_TYPE}_{dtype_case.name}_{forward_case.name}",
        plugin_version=PLUGIN_VERSION,
        plugin_namespace=PLUGIN_NAMESPACE,
    )
    graph = helper.make_graph(
        [node],
        f"ms_deform_attn_{dtype_case.name}_{forward_case.name}",
        [value_info, sampling_info, weight_info],
        [output_info],
        initializer=[
            numpy_helper.from_array(spatial_shapes.cpu().numpy().astype(np.int32), "spatial_shapes"),
            numpy_helper.from_array(level_start_index.cpu().numpy().astype(np.int32), "level_start_index"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 17),
            helper.make_operatorsetid("nvidia", 17),
        ],
        producer_name="pose_msda_plugin_unit_test",
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.save(model, str(onnx_path))


def build_engine(onnx_path: Path, dtype_case: DtypeCase):
    _preload_tensorrt_libs()
    import tensorrt as trt

    _load_plugin_handle()

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    registry = trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD)
    creator = registry.get_creator(PLUGIN_OP_TYPE, PLUGIN_VERSION, PLUGIN_NAMESPACE)
    if creator is None:
        raise RuntimeError(
            f"TensorRT plugin creator not registered: "
            f"{PLUGIN_OP_TYPE}, version={PLUGIN_VERSION}, namespace={PLUGIN_NAMESPACE}"
        )

    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.builder_optimization_level = 3

    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errors}")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"TensorRT engine build failed for {dtype_case.name}")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError(f"TensorRT engine deserialize failed for {dtype_case.name}")
    return engine


class TrtRunner:
    def __init__(
        self,
        engine,
        value: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> None:
        import tensorrt as trt

        self.engine = engine
        self.context = engine.create_execution_context()
        self.inputs = {
            "value": value.contiguous(),
            "sampling_locations": sampling_locations.contiguous(),
            "attention_weights": attention_weights.contiguous(),
        }

        for name, tensor in self.inputs.items():
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())

        missing = self.context.infer_shapes()
        if missing:
            raise RuntimeError(f"TensorRT shape inference has missing tensors: {missing}")

        output_shape = tuple(self.context.get_tensor_shape("output"))
        output_dtype = torch.float16 if engine.get_tensor_dtype("output") == trt.DataType.HALF else torch.float32
        self.output = torch.empty(output_shape, dtype=output_dtype, device=value.device)
        self.context.set_tensor_address("output", self.output.data_ptr())

    def run(self) -> torch.Tensor:
        ok = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        return self.output


def run_engine(engine, value: torch.Tensor, sampling_locations: torch.Tensor, attention_weights: torch.Tensor) -> torch.Tensor:
    runner = TrtRunner(engine, value, sampling_locations, attention_weights)
    output = runner.run()
    torch.cuda.synchronize()
    return output


@torch.no_grad()
def check_forward_equal_with_pytorch(dtype_case: DtypeCase, seed: int, optimized_kernel: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT MSDA plugin unit test")

    forward_case = ForwardCase(
        name="unit",
        batch=1,
        num_query=16,
        num_heads=2,
        channels=32,
        num_points=4,
        spatial_shapes=((4, 4), (2, 2), (2, 1), (1, 1)),
    )
    set_optimized_kernel_enabled(optimized_kernel)
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = make_inputs(
        forward_case, dtype_case.torch_dtype, seed=seed
    )
    reference = ms_deform_attn_core_pytorch(
        value.float(),
        spatial_shapes.to(torch.int64),
        sampling_locations.float(),
        attention_weights.float(),
    )
    if dtype_case.torch_dtype is torch.float16:
        reference = reference.half()

    with tempfile.TemporaryDirectory(prefix=f"pose_msda_{dtype_case.name}_") as tmpdir:
        onnx_path = Path(tmpdir) / f"ms_deform_attn_{dtype_case.name}.onnx"
        build_onnx(onnx_path, dtype_case, forward_case, spatial_shapes, level_start_index)
        engine = build_engine(onnx_path, dtype_case)
        output = run_engine(engine, value, sampling_locations, attention_weights)

    diff = (output.float() - reference.float()).abs()
    max_abs_err = diff.max().item()
    max_rel_err = (diff / reference.float().abs().clamp_min(1e-6)).max().item()
    fwdok = torch.allclose(output.float(), reference.float(), rtol=dtype_case.rtol, atol=dtype_case.atol)

    print(
        f"* {fwdok} check_forward_equal_with_pytorch_{dtype_case.name}_{_kernel_mode_name(optimized_kernel)}: "
        f"max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
    )
    if not fwdok:
        raise AssertionError(
            f"{dtype_case.name} mismatch: max_abs_err={max_abs_err:.6e}, max_rel_err={max_rel_err:.6e}, "
            f"rtol={dtype_case.rtol}, atol={dtype_case.atol}"
        )


def _estimate_io_gbps(tensors: Iterable[torch.Tensor], avg_ms: float) -> float:
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    return total_bytes / (avg_ms / 1e3) / 1e9


def _format_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "FP32"
    if dtype == torch.float16:
        return "FP16"
    return str(dtype).replace("torch.", "").upper()


def _format_case(dtype_case: DtypeCase, forward_case: ForwardCase, optimized_kernel: bool | None = None) -> str:
    kernel_part = "" if optimized_kernel is None else f" K={_kernel_mode_name(optimized_kernel):<9s}"
    return (
        "{dtype}{kernel_part} B={batch:<4d} Q={num_query:<4d} H={num_heads:<2d} "
        "C={channels:<2d} L={num_levels:<1d} P={num_points:<1d} ({name})"
    ).format(
        dtype=_format_dtype(dtype_case.torch_dtype),
        kernel_part=kernel_part,
        batch=forward_case.batch,
        num_query=forward_case.num_query,
        num_heads=forward_case.num_heads,
        channels=forward_case.channels,
        num_levels=forward_case.num_levels,
        num_points=forward_case.num_points,
        name=forward_case.name,
    )


@torch.no_grad()
def benchmark_forward(dtype_case: DtypeCase, forward_case: ForwardCase, warm_iters: int, num_iters: int, seed: int) -> None:
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = make_inputs(
        forward_case, dtype_case.torch_dtype, seed=seed
    )

    with tempfile.TemporaryDirectory(prefix=f"pose_msda_bench_{dtype_case.name}_{forward_case.name}_") as tmpdir:
        onnx_path = Path(tmpdir) / f"ms_deform_attn_{dtype_case.name}_{forward_case.name}.onnx"
        build_onnx(onnx_path, dtype_case, forward_case, spatial_shapes, level_start_index)
        engine = build_engine(onnx_path, dtype_case)
        runner = TrtRunner(engine, value, sampling_locations, attention_weights)

        for optimized_kernel in (False, True):
            set_optimized_kernel_enabled(optimized_kernel)
            for _ in range(warm_iters):
                runner.run()
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(num_iters):
                runner.run()
            end.record()
            torch.cuda.synchronize()

            avg_ms = start.elapsed_time(end) / num_iters
            gbps = _estimate_io_gbps((value, sampling_locations, attention_weights, runner.output), avg_ms)
            print(
                "[BENCH] {case:<76s} {latency_us:>10.2f} us {gbps:>8.2f} GB/s".format(
                    case=_format_case(dtype_case, forward_case, optimized_kernel),
                    latency_us=avg_ms * 1e3,
                    gbps=gbps,
                )
            )


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("TensorRT MSDA plugin correctness and benchmark test.")
    parser.add_argument("--num-iters", type=int, default=100, help="benchmark iterations")
    parser.add_argument("--warm-iters", type=int, default=10, help="benchmark warmup iterations")
    parser.add_argument("--seed", type=int, default=3, help="random seed")
    parser.add_argument("--skip-check", action="store_true", help="skip correctness check")
    parser.add_argument("--skip-bench", action="store_true", help="skip benchmark")
    parser.add_argument(
        "--include-large-batches",
        action="store_true",
        help="also benchmark batch=64/256/1024 cases from the reference benchmark; may require large GPU memory",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_iters <= 0:
        raise ValueError("--num-iters must be positive")
    if args.warm_iters < 0:
        raise ValueError("--warm-iters must be non-negative")


def dtype_cases() -> tuple[DtypeCase, ...]:
    from onnx import TensorProto

    return (
        # DtypeCase("float32", torch.float32, TensorProto.FLOAT, rtol=1e-2, atol=1e-3),
        DtypeCase("half", torch.float16, TensorProto.FLOAT16, rtol=2e-2, atol=2e-3),
    )


def benchmark_cases(include_large_batches: bool = False) -> tuple[ForwardCase, ...]:
    base = ForwardCase(
        name="deformable-detr",
        batch=1,
        num_query=12240,# 9200 for decoder
        num_heads=8,
        channels=32,
        num_points=4,
        spatial_shapes=((96, 96), (48, 48), (24, 24), (12, 12)),
    )
    batches = (1, 2, 4, 16)
    if include_large_batches:
        batches = batches + (64, 256, 1024)
    return tuple(replace(base, name=f"batched-{batch}", batch=batch) for batch in batches)


def main() -> int:
    args = get_arg_parser().parse_args()
    _validate_args(args)
    _preload_tensorrt_libs()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT MSDA plugin tests")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.skip_check:
        print("=== TensorRT MSDA plugin forward checks ===")
        for dtype_case in dtype_cases():
            for optimized_kernel in (False, True):
                check_forward_equal_with_pytorch(dtype_case, seed=args.seed, optimized_kernel=optimized_kernel)

    if not args.skip_bench:
        print(
            "=== TensorRT MSDA plugin forward benchmarks "
            f"(warmup={args.warm_iters}, iters={args.num_iters}) ==="
        )
        for forward_case in benchmark_cases(args.include_large_batches):
            for dtype_case in dtype_cases():
                benchmark_forward(dtype_case, forward_case, args.warm_iters, args.num_iters, seed=args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
