from TFConvertor import OnnxConvertor, TFConvertor, OnnxNodeConvertor, NodeConvertor
import argparse
from TFDL2 import TFDataType, Op, DECODER_FLAGS
from collections import OrderedDict
import numpy as np
import glob
import os
from typing import Dict, List, Union
from pathlib import Path
import random


class InputShapeAction(argparse.Action):
    """自定义 Action 来处理输入形状"""

    def __call__(self, parser, namespace, values, option_string=None):
        result = {}
        for value in values:
            if ":" not in value:
                parser.error(f"无效格式: {value}。请使用 inputname:1,3,224,224 格式")

            name, shape_str = value.split(":", 1)
            name = name.strip()

            try:
                shape_values = [int(x.strip()) for x in shape_str.split(",")]
                if not shape_values:
                    parser.error(f"形状值不能为空: {value}")
                result[name] = shape_values
            except ValueError:
                parser.error(f"无效的形状值: {shape_str}")

        setattr(namespace, self.dest, result)


def from_onnx(args):
    assert args.onnxpath is not None
    destination = Path(args.output_name).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    convertor = OnnxConvertor(destination.name)
    convertor.load(args.onnxpath, stoptensor=args.stoptensor)
    # 这里后面可以使用自定义的模型优化器，比如合并算子，剪枝算子等
    convertor.optmize()  # ([custom1,custom3])#([custom1,custom2])
    # 这里可以注册自定义的节点转换器，比如自定义的节点类型，或者需要特殊处理的节点类型，后面函数的书写方法参考customop.py
    # convertor.registorCustomNodeConvertor("PPScatterPlugin",PillarScatterPlugin)
    # convertor.registorCustomNodeConvertor("Optimize1",Optimize1)

    convertor.buildTFmodel(inputshape=args.input_shapes, std=args.std, mean=args.mean)
    if (
        args.calibration_backend != "sdk-cpu"
        or args.auto_fallback
        or args.calibration_list
        or args.calibration_bin
        or args.validation_bin
    ):
        if args.fp16:
            raise ValueError(
                "--fp16 is incompatible with the FP32-guarded calibration/search path"
            )
        if args.check and not convertor.verification(targetModel=args.testonnxpath):
            raise RuntimeError("ONNX conversion verification failed")
        convertor.dump(str(destination))
        if args.calibration_list:
            base = Path(args.calibration_list).resolve().parent
            imglist = [
                str((base / Path(s.strip()).expanduser()).resolve())
                for s in Path(args.calibration_list).read_text().splitlines()
                if s.strip()
            ]
        elif args.quantimgDir:
            groups = [
                sorted(
                    str(p)
                    for p in d.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                )
                for d in sorted(Path(args.quantimgDir).iterdir())
                if d.is_dir()
            ]
            groups = [g for g in groups if g]
            if not groups:
                groups = [
                    sorted(
                        str(p)
                        for p in Path(args.quantimgDir).rglob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                    )
                ]
            rng = random.Random(args.seed)
            imglist = []
            if args.calibration_count < len(groups):
                raise ValueError("calibration-count must cover all class folders")
            for k, g in enumerate(groups):
                n = args.calibration_count // len(groups) + (
                    k < args.calibration_count % len(groups)
                )
                if len(g) < n:
                    raise ValueError("not enough calibration images")
                imglist.extend(rng.sample(g, n))
            rng.shuffle(imglist)
        else:
            imglist = None
        options = dict(
            source_onnx=args.onnxpath,
            calibration_bin=args.calibration_bin,
            threads=args.calibration_threads,
            workers=args.calibration_workers,
            compile=args.calibration_compile,
            chip=args.chip,
            cuda_device=args.cuda_device,
            cosine_distance=args.cosine_distance,
            min_cosine=args.min_cosine,
            max_evals=args.max_search_evals,
            tolerate=args.tolerate,
            workdir=args.quant_workdir,
            validation_bin=args.validation_bin,
            validation_labels=args.validation_labels,
            require_validation_target=args.require_validation_target,
            max_accuracy_loss=args.max_accuracy_loss,
        )
        convertor.quantContext(
            calibration_list=imglist,
            decoderflags=DECODER_FLAGS(args.cvtype),
            MergeConcate=not args.no_merge_concat,
            calibration_backend=args.calibration_backend,
            auto_fallback=args.auto_fallback,
            calibration_options=options,
        )
        convertor.dump(str(destination) + ".quant")
        if args.auto_fallback or args.validation_bin:
            from TFConvertor.calibration_backend import audit

            audit(
                str(destination.resolve()) + ".quant.fb",
                convertor._quant_workdir,
                options,
                convertor._quant_state,
            )
        print("Quantized model:", str(destination) + ".quant.fb")
        print("Calibration/search report:", convertor._quant_workdir + "/report.json")
        return convertor
    if args.check:
        if convertor.verification(targetModel=args.testonnxpath):
            convertor.dump(str(destination))
            print("Convert Success!!!")
            if args.quantimgDir is not None:
                imglist = glob.glob(os.path.join(args.quantimgDir, "*.jpg"))
                convertor.quantContext(
                    calibration_list=imglist,
                    decoderflags=DECODER_FLAGS(args.cvtype),
                    MergeConcate=False,
                )
            else:
                print("没有量化图片集，使用随机数量化")
                convertor.quantContext(
                    calibration_list=None,
                    decoderflags=DECODER_FLAGS(args.cvtype),
                    MergeConcate=False,
                )
            if args.fp16:
                convertor.toFp16()
            convertor.dump(str(destination) + ".quant")
        else:
            print("Convert Fail!")
    else:
        if args.fp16:
            convertor.toFp16()
        convertor.dump(str(destination))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="from_onnx.py")
    parser.add_argument(
        "--onnxpath", type=str, help="用来转化的onnx模型地址", required=True
    )
    parser.add_argument(
        "--testonnxpath",
        type=str,
        help="如果onnx模型是修改过的无法直接forward,此时使用这个备用onnx模型地址,如果没有修改onnx则不需要指定此参数",
    )
    parser.add_argument("--output_name", type=str, help="导出模型名称", required=True)
    parser.add_argument("--fp16", action="store_true", help="将fp32模型降为半精度")
    parser.add_argument(
        "--stoptensor", type=str, nargs="+", help="模型转化停止节点名称", default=None
    )
    parser.add_argument("--quantimgDir", type=str, help="使用的量化图片集的文件夹地址")
    parser.add_argument(
        "--cvtype",
        type=int,
        help="模型接受的输入图片通道类型 0:BGR 1:RGB 2Gray",
        default=1,
    )
    parser.add_argument("--mean", type=float, nargs="+", help="preproce mean")
    parser.add_argument("--std", type=float, nargs="+", help="preprocess std")
    parser.add_argument(
        "--input_shapes",
        type=str,
        nargs="+",
        action=InputShapeAction,
        help="如果原始onnx模型输入是dynamic的,为了能顺利比较转化是否成功需要手动指定模型的输入尺寸，输入形状，格式: inputname:1,3,224,224 inputname2:1,3,224,224",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="是否比较转化的模型与原始onnx的输出结果,从而判断转化是否成功",
    )
    parser.add_argument(
        "--calibration-backend",
        choices=["sdk-cpu", "torch-cpu", "torch-cuda", "tfllm-npu"],
        default="sdk-cpu",
        help="range collector and auto-fallback reference/candidate evaluator; exported FB is audited by the SDK",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="visible CUDA device index for torch-cuda (separate from --chip)",
    )
    parser.add_argument(
        "--calibration-list", help="one image per line, relative to this file"
    )
    parser.add_argument(
        "--calibration-bin", help="already normalized float32 NCHW samples"
    )
    parser.add_argument("--calibration-count", type=int, default=20)
    parser.add_argument("--calibration-threads", type=int, default=8)
    parser.add_argument(
        "--calibration-workers",
        type=int,
        default=1,
        help="tfllm-npu request threads sharing one model/runtime",
    )
    parser.add_argument("--calibration-compile", action="store_true",
                        help="compile fixed-shape threaded NPU calibration; includes one-time compiler startup")
    parser.add_argument(
        "--auto-fallback",
        action="store_true",
        help="recursive cosine-guided rollback, no model-specific node list",
    )
    parser.add_argument("--cosine-distance", type=float, default=0.002)
    parser.add_argument("--min-cosine", type=float, default=0.90)
    parser.add_argument("--max-search-evals", type=int, default=128)
    parser.add_argument("--tolerate", type=int, default=16,
                        help="stop after this many new candidates without reducing feasible float units; 0 disables")
    parser.add_argument("--quant-workdir")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--chip", type=int, default=0)
    parser.add_argument("--no-merge-concat", action="store_true")
    parser.add_argument(
        "--validation-bin",
        help="held-out normalized FP32 input tensors; never used to guide the search",
    )
    parser.add_argument("--validation-labels")
    parser.add_argument("--require-validation-target", action="store_true")
    parser.add_argument(
        "--max-accuracy-loss",
        type=float,
        help="hard validation accuracy gate, absolute percentage points; requires labels",
    )
    args = parser.parse_args()
    if (
        not 0 <= args.cosine_distance <= 2
        or not -1 <= args.min_cosine <= 1
        or args.max_search_evals < 2
        or args.tolerate < 0
        or args.calibration_threads < 1
        or not 1 <= args.calibration_workers <= 16
        or args.calibration_count < 1
        or args.chip < 0
        or args.cuda_device < 0
    ):
        parser.error("invalid calibration/search parameters")
    if args.calibration_workers != 1 and args.calibration_backend != "tfllm-npu":
        parser.error("multiple calibration workers currently require tfllm-npu")
    if args.calibration_compile and args.calibration_backend != "tfllm-npu":
        parser.error("compiled calibration requires tfllm-npu")
    if (
        args.validation_labels or args.require_validation_target
    ) and not args.validation_bin:
        parser.error("validation options require --validation-bin")
    if args.max_accuracy_loss is not None and (
        not args.validation_labels
        or not np.isfinite(args.max_accuracy_loss)
        or args.max_accuracy_loss < 0
    ):
        parser.error(
            "--max-accuracy-loss requires validation labels and a finite nonnegative value"
        )
    if (
        sum(
            bool(v)
            for v in (args.calibration_bin, args.calibration_list, args.quantimgDir)
        )
        > 1
    ):
        parser.error(
            "choose one calibration source: --calibration-bin, --calibration-list, or --quantimgDir"
        )
    from_onnx(args)
