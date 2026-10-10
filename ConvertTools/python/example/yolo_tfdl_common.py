"""Inspect an exported YOLO graph and convert its actual layout to TFDL.

Ultralytics changes output paths and node numbering across releases. Quantizer
boundaries are therefore selected from ONNX data flow, not model versions or
hard-coded ``/model.23`` names.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class YoloGraphPlan:
    stop_tensor: str | None
    protected_nodes: tuple[str, ...]
    avoid_nodes: tuple[str, ...]
    output_names: tuple[str, ...]
    source_version: str | None


def _dimensions(value) -> tuple[int | None, ...]:
    return tuple(dim.dim_value if dim.HasField("dim_value") else None
                 for dim in value.type.tensor_type.shape.dim)


def inspect_graph(path: str | Path) -> YoloGraphPlan:
    """Select conversion boundaries from ONNX topology and output signatures."""
    model = onnx.load_model(str(path), load_external_data=False)
    graph = model.graph
    if not graph.output:
        raise ValueError("ONNX has no graph output")
    metadata = {item.key: item.value for item in model.metadata_props}
    producers = {tensor: node for node in graph.node for tensor in node.output}
    consumers: dict[str, list] = {}
    for node in graph.node:
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(node)

    def ancestors(tensors: list[str]) -> set[int]:
        result: set[int] = set()
        pending = list(tensors)
        while pending:
            node = producers.get(pending.pop())
            if node is None or id(node) in result:
                continue
            result.add(id(node))
            pending.extend(node.input)
        return result

    first_shape = _dimensions(graph.output[0])
    if len(first_shape) != 3:
        raise ValueError(f"Expected a rank-3 YOLO detection output, got {first_shape}")
    # End-to-end segmentation adds mask coefficients to each row, so its last
    # dimension is 6+nm rather than 6. Detect the selection tail itself.
    first_ancestors = ancestors([graph.output[0].name])
    processed = metadata.get("end2end", "").lower() == "true" or any(
        id(node) in first_ancestors and node.op_type in {"TopK", "NonMaxSuppression"}
        for node in graph.node
    )
    stop = None
    if processed:
        # Detection splits box/class channels; segmentation slices box, class,
        # and mask coefficients. Both occur immediately after this Transpose.
        first_selection = min((index for index, node in enumerate(graph.node)
                               if id(node) in first_ancestors
                               and node.op_type in {"TopK", "NonMaxSuppression"}),
                              default=len(graph.node))
        candidates = [node.output[0] for index, node in enumerate(graph.node)
                      if index < first_selection and id(node) in first_ancestors
                      and node.op_type == "Transpose" and node.output
                      and (any(user.op_type == "Split" for user in consumers.get(node.output[0], ()))
                           or sum(user.op_type == "Slice" for user in consumers.get(node.output[0], ())) >= 2)]
        if not candidates:
            raise ValueError(
                "Cannot find the raw detection tensor before YOLO postprocessing. "
                "Re-export with nms=None, simplify=False."
            )
        stop = candidates[-1]

    # Only inspect nodes needed for the actual TFDL outputs. An ONNX file may
    # also contain dead branches left by export or graph simplification.
    targets = [stop or graph.output[0].name, *(item.name for item in graph.output[1:])]
    required = ancestors(targets)

    protected = []
    avoid = []
    for node in graph.node:
        if id(node) not in required or not node.output:
            continue
        source = producers.get(node.input[0]) if node.input else None
        if (node.op_type == "Reshape" and source is not None and source.op_type == "Conv"
                and any(user.op_type == "Concat" for user in consumers.get(node.output[0], ()))):
            protected.append(node.name)
        if node.op_type == "Conv" and source is not None and source.op_type == "Transpose":
            before = producers.get(source.input[0]) if source.input else None
            if before is not None and before.op_type == "Softmax":
                avoid.append(node.name)  # DFL projection

    for name in (*protected, *avoid):
        if not name:
            raise ValueError("ONNX detection-head nodes need names for TFDL quantization")
    if len(protected) != len(set(protected)) or len(avoid) != len(set(avoid)):
        raise ValueError("ONNX detection-head node names are not unique")
    return YoloGraphPlan(stop, tuple(protected), tuple(avoid),
                         tuple(targets), metadata.get("version"))


class InputShapeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        shapes = {}
        for value in values:
            try:
                name, dimensions = value.split(":", 1)
                shape = [int(part) for part in dimensions.split(",")]
                if not name or not shape or any(d <= 0 for d in shape):
                    raise ValueError
            except ValueError:
                parser.error(f"Invalid input shape {value!r}; use images:1,3,640,640")
            shapes[name] = shape
        setattr(namespace, self.dest, shapes)


def build_parser(family: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Convert Ultralytics {family} ONNX to TFDL")
    parser.add_argument("--onnxpath", required=True)
    parser.add_argument("--testonnxpath", help="ONNX file for numerical verification, if different")
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--input_shapes", nargs="+", action=InputShapeAction)
    parser.add_argument("--quantimgDir")
    parser.add_argument("--cvtype", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--mean", type=float, nargs="+")
    parser.add_argument("--std", type=float, nargs="+")
    parser.add_argument("--detectOnly", action="store_true", help="Kept for existing commands; layout is detected from ONNX")
    parser.add_argument("--check", action="store_true", help="Verify ONNX against TFDL, then quantize")
    parser.add_argument("--fp16", action="store_true")
    return parser


def convert(args: argparse.Namespace) -> YoloGraphPlan:
    from TFDL2 import DECODER_FLAGS
    from TFConvertor import OnnxConvertor

    plan = inspect_graph(args.onnxpath)
    print(f"YOLO graph: Ultralytics {plan.source_version or 'unknown'}, "
          f"outputs={plan.output_names}, protected={len(plan.protected_nodes)}, "
          f"DFL={len(plan.avoid_nodes)}", flush=True)
    converter = OnnxConvertor(args.output_name)
    converter.load(args.onnxpath, stoptensor=[plan.stop_tensor] if plan.stop_tensor else None)
    converter.optmize()
    converter.buildTFmodel(inputshape=args.input_shapes, std=args.std, mean=args.mean)

    if args.check:
        if not converter.verification(targetModel=args.testonnxpath or args.onnxpath):
            raise RuntimeError("YOLO ONNX/TFDL numerical verification failed")
    if args.fp16 and not args.check:
        converter.toFp16()
    converter.dump(args.output_name)

    if args.check:
        if args.quantimgDir is None:
            images = None  # Existing CLI behavior: random calibration.
            print("No calibration images supplied; using random calibration")
        else:
            directory = Path(args.quantimgDir)
            images = sorted(str(path) for path in directory.iterdir()
                            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if not images:
                raise ValueError(f"No calibration images found in {directory}")
        converter.quantContext(
            calibration_list=images, decoderflags=DECODER_FLAGS(args.cvtype),
            MergeConcate=False, avoidtensors=plan.avoid_nodes,
            stopquanttensors=plan.protected_nodes,
        )
        if args.fp16:
            converter.toFp16()
        converter.dump(f"{args.output_name}.quant")
    return plan


def main(family: str) -> None:
    convert(build_parser(family).parse_args())
