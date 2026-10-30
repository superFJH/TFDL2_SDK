"""Budgeted recursive precision search; no assumption of additive layer errors."""

from collections import deque
from .metrics import passes


class RecursiveRollback:
    """Keep a feasible incumbent; split failed quantization groups recursively.

    No monotonic-error assumption is made. A finite budget gives the best
    measured feasible candidate, not a proof of a globally optimal layer set.
    """

    def __init__(self, units, evaluate, target=0.002, min_cosine=0.90,
                 max_evals=128, score=None, tolerate=16, on_result=None):
        if max_evals < 2 or tolerate < 0:
            raise ValueError("max_evals must be >= 2 and tolerate must be >= 0")
        self.units = list(units)
        self.evaluate = evaluate
        self.target = target
        self.min_cosine = min_cosine
        self.max_evals = max_evals
        self.cache = {}
        self.trace = []
        self.best = None
        self.score = score or (lambda row: (row["mean_ms"], len(row["selected"])))
        self.tolerate = tolerate
        self.on_result = on_result
        self.best_float_units = None
        self.stalled_evaluations = 0
        self.stop_reason = None

    def test(self, selected, phase):
        key = tuple(sorted(selected))
        if key in self.cache:
            return self.cache[key]
        if len(self.cache) >= self.max_evals:
            return None
        m = self.evaluate(set(selected), len(self.cache))
        row = dict(m, selected=list(key), phase=phase, evaluation=len(self.cache))
        row["satisfied"] = passes(row, self.target, self.min_cosine)
        self.cache[key] = row
        self.trace.append(row)
        if row["satisfied"] and (
            self.best is None
            or self.score(row) < self.score(self.best)
        ):
            self.best = row
        improved = row["satisfied"] and (
            self.best_float_units is None or len(key) < self.best_float_units
        )
        if improved:
            self.best_float_units = len(key)
            self.stalled_evaluations = 0
        elif phase == "requantize-or-rollback":
            self.stalled_evaluations += 1
        row.update(best_float_units=self.best_float_units,
                   stalled_evaluations=self.stalled_evaluations)
        if self.on_result is not None:
            self.on_result(row)
        return row

    def run(self):
        baseline = self.test(set(), "int8-baseline")
        if baseline["satisfied"]:
            self.stop_reason = "baseline_satisfied"
            return self.result(set())
        current = set(self.units)
        root = self.test(current, "float-guard")
        if root is None or not root["satisfied"]:
            self.stop_reason = "float_guard_failed"
            return self.result(current)
        pending = deque([self.units])
        while pending and len(self.cache) < self.max_evals:
            group = pending.popleft()
            if not group:
                continue
            candidate = current - set(group)
            row = self.test(candidate, "requantize-or-rollback")
            if row is None:
                break
            if row["satisfied"]:
                current = candidate
            elif len(group) > 1:
                mid = len(group) // 2
                pending.extend([group[:mid], group[mid:]])
            if self.tolerate and self.stalled_evaluations >= self.tolerate:
                self.stop_reason = "tolerate"
                break
        if self.stop_reason is None:
            self.stop_reason = "max_evals" if pending and len(self.cache) >= self.max_evals else "completed"
        return self.result(current)

    def result(self, current):
        return dict(
            satisfied=self.best is not None,
            best=self.best,
            current_selected=sorted(current),
            evaluations=len(self.cache),
            budget_exhausted=len(self.cache) >= self.max_evals,
            early_stopped=self.stop_reason == "tolerate",
            stop_reason=self.stop_reason,
            tolerate=self.tolerate,
            stalled_evaluations=self.stalled_evaluations,
            best_float_units=self.best_float_units,
            target_distance=self.target,
            min_cosine=self.min_cosine,
            trace=self.trace,
        )
