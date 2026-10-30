"""SDK-native automatic quantization. Optional Torch imports stay lazy."""

import hashlib
import json
import tempfile
import time
from pathlib import Path
import numpy as np
from .metrics import compare, passes, classification
from .search import RecursiveRollback
from .evaluation import create_evaluator, sdk_config


def read_images(graph, paths, color, normalized_bin=None):
    if normalized_bin:
        a = np.fromfile(normalized_bin, dtype="<f4")
        per = int(np.prod(graph.shape))
        if not a.size or a.size % per:
            raise ValueError("input binary does not match NCHW shape")
        a = a.reshape(-1, *graph.shape[1:])
    else:
        import cv2

        frames = []
        channels = graph.shape[1]
        if color == 2 and channels != 1:
            raise ValueError("gray decoder requires C=1")
        for path in paths or []:
            image = cv2.imread(
                str(path), cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
            )
            if image is None:
                raise ValueError("cannot read " + str(path))
            if channels == 3 and color == 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(
                image, (graph.shape[3], graph.shape[2]), interpolation=cv2.INTER_LINEAR
            )
            if channels == 1:
                image = image[:, :, None]
            frames.append(
                ((image.transpose(2, 0, 1)[None] - graph.mean) * graph.scale)[0]
            )
        if not frames:
            raise ValueError("real calibration images/tensors are required")
        a = np.asarray(frames, np.float32)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite input")
    return np.ascontiguousarray(a, dtype=np.float32)


def quantize(
    converter,
    calibration_list,
    quant_input,
    decoderflags,
    merge_concat,
    avoids,
    stops,
    backend,
    auto_fallback,
    options,
):
    from TFDL2 import TFDataType as DT
    from .graph import Graph, sha256

    options = dict(options or {})
    if getattr(converter, "_automatic_quantized", False):
        raise ValueError(
            "rebuild the original floating converter before quantizing again"
        )
    target = float(options.get("cosine_distance", 0.002))
    minimum = float(options.get("min_cosine", 0.90))
    if (
        not 0 <= target <= 2
        or not -1 <= minimum <= 1
        or int(options.get("max_evals", 128)) < 2
        or int(options.get("tolerate", 16)) < 0
        or not 1 <= int(options.get("threads", 8)) <= 256
        or int(options.get("chip", 0)) < 0
        or int(options.get("cuda_device", 0)) < 0
        or not 1 <= int(options.get("workers", 1)) <= 16
    ):
        raise ValueError("invalid calibration/search options")
    if int(options.get("workers", 1)) != 1 and backend != "tfllm-npu":
        raise ValueError("multiple calibration workers currently require tfllm-npu")
    executor = options.get("executor", "threads" if backend == "tfllm-npu" else "processes")
    if executor not in ("processes", "threads"):
        raise ValueError("calibration executor must be processes or threads")
    if options.get("compile", False) and (
        backend != "tfllm-npu" or executor != "threads"
    ):
        raise ValueError("compiled calibration requires tfllm-npu with the threads executor")
    if executor == "threads" and backend != "tfllm-npu":
        raise ValueError("the threads calibration executor requires tfllm-npu")
    if options.get("sdk_lib") or options.get("sdk_root"):
        raise ValueError(
            "use the installed SDK Python package and its matching runtime; SDK library overrides are no longer supported"
        )
    limit = options.get("max_accuracy_loss")
    if limit is not None and (
        not options.get("validation_labels")
        or not options.get("validation_bin")
        or not np.isfinite(limit)
        or limit < 0
    ):
        raise ValueError(
            "accuracy gate requires validation inputs, labels and a finite nonnegative limit"
        )
    if options.get("require_validation_target") and not options.get("validation_bin"):
        raise ValueError("validation gate requires validation inputs")
    started = time.perf_counter()
    work = (
        Path(options["workdir"]).resolve()
        if options.get("workdir")
        else Path(tempfile.mkdtemp(prefix="onnx2tfdl-quant-"))
    )
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        raise ValueError("quantization workdir must be empty")
    converter._quant_workdir = str(work)
    report = dict(
        status="started",
        implementation="sdk-python-op",
        backend=backend,
        auto_fallback=bool(auto_fallback),
        quantization_mode="mixed-search" if auto_fallback else "sdk-standard",
        approximate_calibration=backend == "tfllm-npu",
    )

    def save():
        (work / "report.json").write_text(json.dumps(report, indent=2))

    save()
    evaluator = None
    try:
        graph = Graph(converter, int(options.get("threads", 8)))
        images = read_images(
            graph,
            calibration_list,
            int(getattr(decoderflags, "value", decoderflags) or 0),
            options.get("calibration_bin"),
        )
        images.tofile(work / "calibration.f32")
        report.update(
            samples=len(images),
            input_shape=graph.shape,
            calibration_sha256=sha256(work / "calibration.f32"),
            original_units=len(graph.units),
        )
        if options.get("source_onnx"):
            report["source_onnx_sha256"] = sha256(options["source_onnx"])
        if calibration_list:
            report["calibration_samples"] = [
                dict(path=str(Path(p).resolve()), sha256=sha256(p))
                for p in calibration_list
            ]
        converter.dump(str(work / "source-float"))
        if backend == "sdk-cpu":
            ranges = graph.collect_cpu(images)
        elif backend in ("tfllm-npu", "torch-cpu", "torch-cuda"):
            from .torch_backend import collect

            ranges, report["collector"] = collect(
                graph,
                images,
                backend,
                int(options.get("threads", 8)),
                options.get("spatial", "native"),
                int(options.get("chip", 0)),
                int(options.get("cuda_device", 0)),
                workers=int(options.get("workers", 1)),
                workdir=work,
                images_path=work / "calibration.f32",
                executor=executor,
                compile_model=bool(options.get("compile", False)),
            )
        else:
            raise ValueError("unknown calibration backend")

        # Without auto_fallback, retain the original SDK Quantize path and
        # its avoid/stop semantics regardless of the selected collector.
        def mapped(names):
            return {str(converter.nodeMap.get(n, n)) for n in names}

        avoid_units, stop_units = mapped(avoids), mapped(stops)
        if (avoid_units | stop_units) - set(graph.meta):
            raise ValueError("unknown avoid/stop node")
        input_types = {graph.input_name: DT.TFDL_UINT8}
        if quant_input:
            input_types = {
                str(converter.nodeMap.get(n, n)): dtype
                for n, dtype in quant_input.items()
            }
            if set(input_types) != {graph.input_name} or next(
                iter(input_types.values())
            ) not in (DT.TFDL_FLOAT, DT.TFDL_UINT8):
                raise ValueError("expected the model input with type FP32 or UINT8")
        if auto_fallback:
            graph.set_ranges(ranges, merge_concat)
        else:
            graph.register_ranges(ranges)
            graph.calibration.Quantize(
                dict(input_types),
                avoidtensors=tuple(sorted(avoid_units)),
                stopquanttensors=tuple(sorted(stop_units)),
                MergeConcate=merge_concat,
            )
        (work / "ranges.json").write_text(json.dumps(graph.ranges, indent=2))
        reference = None
        if auto_fallback or options.get("validation_bin"):
            evaluator = create_evaluator(graph, backend, options)
            report["evaluator"] = evaluator.report
            reference, _ = evaluator.evaluate(images, quantized=False)
            np.savez(work / "reference.npz", *reference)
        # FB acceptance must execute the actual SDK export. Non-SDK search
        # backends simulate Q/DQ and cannot directly execute an SDK FB file.
        config = sdk_config(backend, options)
        (work / "run.json").write_text(json.dumps(config, indent=2))
        from TFDL2 import TFDL2 as native

        report["sdk_python_sha256"] = sha256(native.__file__)
        converter._quant_state = dict(
            graph=graph,
            images=images,
            reference=reference,
            config=config,
            options=options,
            backend=backend,
        )
        if not auto_fallback:
            converter._context = graph.context
            converter._context.Dump(str(work / "model.quant"))
            report.update(
                status="quantized",
                manual_avoids=sorted(avoid_units),
                manual_stops=sorted(stop_units),
                seconds=time.perf_counter() - started,
            )
            converter._automatic_quantized = True
            save()
            return report
        target = float(options.get("cosine_distance", 0.002))
        minimum = float(options.get("min_cosine", 0.90))

        required = avoid_units
        if required - set(graph.units) or stop_units - set(graph.units):
            raise ValueError("unknown avoid/stop node")
        # A stop boundary protects all descendants, just like the SDK route.
        protected = set()
        frontier = set(stop_units)
        for call in graph.calls:
            from .graph import refs

            if frontier.intersection(refs(call["args"])):
                protected.update(call["outputs"])
                frontier.update(call["outputs"])
        required |= (protected & set(graph.units)) | graph.forced_float
        report["required_floating_units"] = sorted(required)
        input_type = next(iter(input_types.values()))
        simulated = evaluator.report["approximate"]
        report["search_objective"] = "floating_units_then_backend_ms" if simulated else "backend_ms_then_floating_units"

        def candidate_score(metrics, count):
            ms = metrics.get("mean_ms", float("inf"))
            # Fake Q/DQ costs do not predict SDK integer-kernel latency. Avoid
            # picking the floating guard merely because it has fewer Q/DQs.
            return (count, ms) if simulated else (ms, count)

        def evaluate(selected, index):
            selected = set(selected) | required
            try:
                outputs, ms = evaluator.evaluate(images, selected, input_type=input_type)
                m = compare(reference, outputs, ms)
            except FloatingPointError as error:
                m = dict(valid=False, reason=str(error))
            return m

        def record(row):
            selected = set(row["selected"]) | required
            row = dict(
                row,
                backend=backend,
                float_units=len(selected),
                selected=sorted(selected),
                best_float_units=(row["best_float_units"] + len(required)
                                  if row["best_float_units"] is not None else None),
            )
            (work / f"eval-{row['evaluation']:04d}.json").write_text(json.dumps(row, indent=2))
            print(
                json.dumps({k: v for k, v in row.items() if k != "selected"}),
                flush=True,
            )

        engine = RecursiveRollback(
            [u for u in graph.units if u not in required],
            evaluate,
            target,
            minimum,
            int(options.get("max_evals", 128)),
            score=lambda row: candidate_score(row, len(row["selected"]) + len(required)),
            tolerate=int(options.get("tolerate", 16)),
            on_result=record,
        )
        result = engine.run()
        report["search"] = result
        print(json.dumps(dict(search_stopped=result["stop_reason"],
                              evaluations=result["evaluations"],
                              best_float_units=(result["best_float_units"] + len(required)
                                                if result["best_float_units"] is not None else None))), flush=True)
        if not result["satisfied"]:
            raise RuntimeError("cosine target not met, including floating guard")
        chosen = sorted(set(result["best"]["selected"]) | required)
        final = graph.build(chosen, input_type=input_type)
        converter._context = final
        final.Dump(str(work / "model.quant"))
        (work / "profile.json").write_text(
            json.dumps(
                dict(
                    floating_units=chosen,
                    weight_precision="fp32",
                    unit="recorded SDK Op",
                ),
                indent=2,
            )
        )
        report.update(
            status="calibration_passed",
            floating_units=len(chosen),
            seconds=time.perf_counter() - started,
        )
        converter._quant_state["selected"] = chosen
        converter._automatic_quantized = True
        save()
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        save()
        raise
    finally:
        if evaluator is not None:
            evaluator.close()


def audit(model, work, options):
    """Reload the official export, then independently accept/reject held-out data."""
    from TFDL2 import TFContext
    from .graph import sha256

    work = Path(work)
    report = json.loads((work / "report.json").read_text())
    # The caller keeps the graph/normalized inputs alive; no FB reader needed.
    state = options["_state"]
    graph = state["graph"]
    backend = state.get("backend", report["backend"])
    evaluator = None
    target = float(options.get("cosine_distance", 0.002))
    minimum = float(options.get("min_cosine", 0.90))
    report.update(
        exported_model=str(Path(model).resolve()), exported_model_sha256=sha256(model)
    )
    try:
        context = TFContext(path=str(model))
        if state["reference"] is None:
            evaluator = create_evaluator(graph, backend, options)
            state["reference"], _ = evaluator.evaluate(state["images"], quantized=False)
        actual, ms = graph.evaluate(context, state["images"], state["config"])
        m = compare(state["reference"], actual, ms)
        m["satisfied"] = passes(m, target, minimum)
        m["backend"] = "sdk-npu" if state["config"].get("UseHardware") else "sdk-cpu"
        report["export_check"] = m
        np.savez(work / "export-check.npz", *actual)
        if not m.get("valid") or not m.get("integer_equal", True):
            raise RuntimeError(
                "exported model produced invalid or incompatible outputs; "
                f"see {work / 'report.json'}"
            )
        m.update(
            mean_cosine=1.0 - m["mean_distance"],
            target_distance=target,
            target_min_cosine=minimum,
            distance_excess=max(0.0, m["mean_distance"] - target),
            min_cosine_shortfall=max(0.0, minimum - m["min_cosine"]),
        )
        best = (report.get("search") or {}).get("best") or {}
        if "mean_distance" in best and "min_cosine" in best:
            m.update(
                search_mean_distance=best["mean_distance"],
                search_min_cosine=best["min_cosine"],
                mean_distance_delta_from_search=m["mean_distance"] - best["mean_distance"],
                min_cosine_delta_from_search=m["min_cosine"] - best["min_cosine"],
            )
        report["status"] = "export_passed" if m["satisfied"] else "exported_target_not_met"
        print(json.dumps(dict(stage="export-check", **m)), flush=True)
        if not m["satisfied"]:
            print(
                "Warning: exported FB did not meet the cosine search target; "
                "the model is retained and conversion continues. "
                f"See actual metrics and deviations in {work / 'report.json'}",
                flush=True,
            )
        if options.get("validation_bin"):
            val = read_images(graph, None, 0, options["validation_bin"])
            hashes = {hashlib.sha256(a.tobytes()).digest() for a in state["images"]}
            overlap = sum(hashlib.sha256(a.tobytes()).digest() in hashes for a in val)
            if evaluator is None:
                evaluator = create_evaluator(graph, backend, options)
            reference, _ = evaluator.evaluate(val, quantized=False)
            actual, ms = graph.evaluate(context, val, state["config"])
            m = compare(reference, actual, ms)
            np.savez(work / "validation-reference.npz", *reference)
            np.savez(work / "validation-actual.npz", *actual)
            m.update(
                samples=len(val),
                overlap_samples=overlap,
                input_sha256=sha256(options["validation_bin"]),
                satisfied=passes(m, target, minimum),
            )
            if options.get("validation_labels"):
                labels = json.loads(Path(options["validation_labels"]).read_text())
                if isinstance(labels, dict):
                    labels = [r["label_alphabetical"] for r in labels["samples"]]
                m.update(classification(reference, actual, labels))
            limit = options.get("max_accuracy_loss")
            if limit is not None:
                m["meets_accuracy_target"] = (
                    m["accuracy_loss_pp"] <= float(limit) + 1e-12
                )
            report["validation"] = m
            if (not m["satisfied"] or not m.get("meets_accuracy_target", True)) and report["export_check"]["satisfied"]:
                report["status"] = "calibration_passed_validation_failed"
            if (
                options.get("require_validation_target")
                and (not m["satisfied"] or overlap)
            ) or (limit is not None and (not m["meets_accuracy_target"] or overlap)):
                report["status"] = "validation_target_not_met"
                raise RuntimeError("held-out cosine/accuracy/independence gate failed")
        return report
    except Exception as error:
        report["error"] = str(error)
        if report["status"] != "validation_target_not_met":
            report["status"] = "failed"
        raise
    finally:
        try:
            if evaluator is not None:
                evaluator.close()
        finally:
            (work / "report.json").write_text(json.dumps(report, indent=2))
