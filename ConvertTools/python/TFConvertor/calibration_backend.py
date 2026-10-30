"""SDK-native calibration, precision search and export acceptance."""

from .quantization import quantize
from .quantization import audit as _audit


def audit(model, work, options, state):
    return _audit(model, work, dict(options, _state=state))
