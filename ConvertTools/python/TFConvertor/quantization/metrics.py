"""Output metrics and held-out acceptance; independent of SDK bindings."""

import numpy as np


def compare(reference, actual, mean_ms=0.0):
    if len(reference) != len(actual):
        raise ValueError("output counts differ")
    cosines = []
    rows = []
    integer_equal = True
    for index, (x, y) in enumerate(zip(reference, actual)):
        if x.shape != y.shape or not len(x):
            raise ValueError("output/sample shapes differ")
        if np.issubdtype(x.dtype, np.integer):
            equal = np.issubdtype(y.dtype, np.integer) and bool(np.array_equal(x, y))
            integer_equal &= equal
            rows.append(dict(output=index, integer_equal=equal))
            continue
        if not np.issubdtype(y.dtype, np.floating):
            raise ValueError("floating output changed to an integer type")
        x = x.reshape(len(x), -1).astype(np.float64)
        y = y.reshape(len(y), -1).astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return dict(valid=False, reason="nonfinite output")
        nx = np.linalg.norm(x, axis=1)
        ny = np.linalg.norm(y, axis=1)
        den = nx * ny
        c = np.zeros(len(x))
        c[(nx == 0) & (ny == 0)] = 1
        good = den > 0
        c[good] = np.clip(np.sum(x[good] * y[good], axis=1) / den[good], -1, 1)
        cosines.extend(c.tolist())
        rows.append(
            dict(
                output=index,
                mean_cosine=float(c.mean()),
                min_cosine=float(c.min()),
                top1_agreement=int(np.sum(x.argmax(1) == y.argmax(1))),
            )
        )
    if not cosines:
        raise ValueError("cosine search needs a floating output")
    return dict(
        valid=True,
        integer_equal=integer_equal,
        mean_distance=float(np.mean(1 - np.asarray(cosines))),
        min_cosine=min(cosines),
        mean_ms=mean_ms,
        outputs=rows,
    )


def passes(metrics, target_distance=0.002, min_cosine=0.90):
    return bool(
        metrics.get("valid")
        and metrics.get("integer_equal", True)
        and metrics["mean_distance"] <= target_distance + 1e-12
        and metrics["min_cosine"] >= min_cosine - 1e-12
    )


def classification(reference, actual, labels):
    if len(reference) != 1 or not np.issubdtype(reference[0].dtype, np.floating):
        raise ValueError("labels require one floating classification output")
    x = reference[0].reshape(len(reference[0]), -1)
    y = actual[0].reshape(x.shape)
    labels = np.asarray(labels)
    if (
        labels.shape != (len(x),)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any(labels < 0)
        or np.any(labels >= x.shape[1])
    ):
        raise ValueError("invalid labels/class order")
    r = float(np.mean(x.argmax(1) == labels))
    a = float(np.mean(y.argmax(1) == labels))
    return dict(accuracy=a, reference_accuracy=r, accuracy_loss_pp=(r - a) * 100)
