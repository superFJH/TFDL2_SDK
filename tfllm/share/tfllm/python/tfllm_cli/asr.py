"""File decoding and Qwen3-ASR prompting; all model inference stays in C++."""
import base64
import contextlib
import io
import math
import os
from pathlib import Path
import re
import sys


LANGUAGES = dict(zip(
    "zh en yue ar de fr es pt id it ko ru th vi ja tr hi ms nl sv da fi pl cs tl fa el ro hu mk".split(),
    "Chinese English Cantonese Arabic German French Spanish Portuguese Indonesian Italian Korean Russian Thai Vietnamese Japanese Turkish Hindi Malay Dutch Swedish Danish Finnish Polish Czech Filipino Persian Greek Romanian Hungarian Macedonian".split()))
_NAMES = {name.lower(): name for name in LANGUAGES.values()}


def language_name(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("language must be a language name or ISO code")
    value = value.strip().lower()
    if value in {"", "auto"}:
        return None
    result = LANGUAGES.get(value) or _NAMES.get(value)
    if not result:
        raise ValueError("Unsupported language: " + value + "; supported ISO codes: " + " ".join(LANGUAGES))
    return result


def audio_dependencies():
    """Check file decoding dependencies before an ASR service becomes ready."""
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as e:
        raise RuntimeError(f"Install audio dependencies: {sys.executable} -m pip install "
                           "'numpy>=1.26,<3' 'soundfile>=0.13,<1' 'scipy>=1.11,<2'"
                           f" (import failed: {e})") from e
    return np, sf, resample_poly


def decode_audio(data, max_seconds=120):
    """Decode bounded audio, average channels, and resample to mono 16 kHz."""
    np, sf, resample_poly = audio_dependencies()
    if not isinstance(data, bytes) or not data or len(data) > 32 * 1024 * 1024:
        raise ValueError("Audio file must contain 1 byte to 32 MiB")
    try:
        with sf.SoundFile(io.BytesIO(data)) as file:
            if not 8000 <= file.samplerate <= 192000 or not 1 <= file.channels <= 16:
                raise ValueError("Unsupported audio sample rate/channels")
            if not file.frames or file.frames > max_seconds * file.samplerate:
                raise ValueError(f"Audio duration must be greater than zero and at most {max_seconds} seconds")
            rate = file.samplerate
            samples = file.read(dtype="float32", always_2d=True).mean(axis=1)
    except (sf.LibsndfileError, RuntimeError) as e:
        raise ValueError("Cannot decode audio file: " + str(e)) from e
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains non-finite samples")
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, rate // divisor)
    samples = np.ascontiguousarray(samples, dtype="<f4")
    if samples.size <= 200 or samples.size > max_seconds * 16000 or not np.isfinite(samples).all():
        raise ValueError("Invalid resampled audio")
    return dict(encoding="f32_mono", sample_rate=16000, frames=int(samples.size),
                samples=base64.b64encode(samples.tobytes()).decode()), samples.size / 16000


def local_audio(path):
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= 32 * 1024 * 1024:
        raise ValueError("Audio file must exist and contain 1 byte to 32 MiB: " + str(path))
    return path.read_bytes()


def parse_transcript(raw, forced_language=None):
    text = re.sub(r"<\|(?:im_end|endoftext)\|>", "", raw).strip()
    if "<asr_text>" in text:
        meta, text = text.split("<asr_text>", 1)
        match = re.search(r"language\s+([A-Za-z]+)", meta)
        detected = match.group(1) if match else ""
        if detected.lower() == "none":
            return "", ""
        detected = _NAMES.get(detected.lower(), "")
        return forced_language or detected, text.strip()
    return forced_language or "", text


class Transcriber:
    def __init__(self, engine, chat):
        if engine.info.get("architecture") != "qwen3-asr" or "audio" not in engine.info.get("modalities", []):
            raise ValueError("Audio transcription requires Qwen3-ASR with its matching audio mmproj")
        self.engine, self.chat = engine, chat
        self.max_seconds = engine.info["audio"]["max_seconds"]

    def request(self, data, language=None, prompt="", temperature=None):
        language = language_name(language)
        if not isinstance(prompt, str) or len(prompt.encode()) > 16384:
            raise ValueError("prompt must contain at most 16384 UTF-8 bytes")
        if temperature is not None and (type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2):
            raise ValueError("temperature must be a number in 0..2")
        audio, duration = decode_audio(data, self.max_seconds)
        values = dict(self.chat.kwargs, messages=[dict(role="system", content=prompt),
                      dict(role="user", content=[dict(type="audio")])],
                      bos_token=self.engine.info["bos_token"], eos_token=self.engine.info["eos_token"],
                      add_generation_prompt=True)
        rendered = self.chat.template.render(**values)
        if rendered.count("<|audio_pad|>") != 1:
            raise ValueError("ASR chat template must render exactly one <|audio_pad|>")
        if language:
            rendered += "language " + language + "<asr_text>"
        request = dict(self.chat.defaults, prompt=rendered, audio=audio, add_special=False,
                       skip_special_tokens=False, eos_token_ids=self.chat.eos)
        if temperature is not None:
            request["temperature"] = temperature
        return request, duration, language

    def transcribe(self, data, language=None, prompt="", temperature=None, cancelled=lambda: False):
        request, duration, forced = self.request(data, language, prompt, temperature)
        result = self.engine.generate(request, cancelled=cancelled)
        if "error" in result or result.get("finish_reason") == "cancelled":
            return result
        detected, text = parse_transcript(result["text"], forced)
        return dict(result, text=text, language=detected, duration=duration, task="transcribe")


def response_body(result, response_format):
    if response_format == "text":
        return result["text"]
    if response_format == "json":
        return dict(text=result["text"])
    if response_format == "verbose_json":
        return {k: result[k] for k in ("task", "language", "duration", "text")}
    raise ValueError("response_format must be json, text or verbose_json; timestamps are not supported")


@contextlib.contextmanager
def native_stdout_to_stderr():
    """Keep C++/NPU diagnostic output out of CLI transcript/JSON stdout."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        # Flush native stdio before restoring stdout (including CPU test logs).
        import ctypes
        ctypes.CDLL(None).fflush(None)
        os.dup2(saved, 1)
        os.close(saved)
