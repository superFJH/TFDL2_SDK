from typing import Callable

from . import TFDL2


def LoadCustomOp(path: str):
    result = TFDL2.LoadCustomOp(path)
    # RegisterCustomOpFromFile follows the C API convention used by the C++
    # frontend: zero is success.  Older wrappers returned None, which remains
    # accepted so a source-only deployment update does not break immediately.
    if result not in (None, 0):
        raise RuntimeError(
            f"failed to register TFDL2 custom-op library {path!r}: {result}"
        )
    return result


def CustomReshape(*args, **kwargs):
    pass


def CustomEval(*args, **kwargs):
    pass


def RegisterCustomOp(OpName: str, Reshape: Callable, Eval: Callable):
    TFDL2.RegisterCustomOp(OpName, Reshape, Eval)
