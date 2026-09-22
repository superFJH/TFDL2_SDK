"""Bounded video transport/decoding. Model-specific preparation is separate."""
import base64
import io
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

MAX_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 120


def local_video(path):
    path = Path(path).expanduser()
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Video exceeds 20 MiB")
    return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def read_video(part):
    if set(part) != {"type", "video_url"} or not isinstance(part["video_url"], dict) or set(part["video_url"]) != {"url"}:
        raise ValueError("Expected video_url: {url: HTTP(S) or base64 data:video URL}")
    url = part["video_url"]["url"]
    if not isinstance(url, str):
        raise ValueError("Video URL must be a string")
    if url.startswith("data:video/"):
        head, sep, data = url.partition(",")
        if not sep or not head.endswith(";base64") or len(data) > 4*((MAX_BYTES+2)//3):
            raise ValueError("Invalid video data URL or video exceeds 20 MiB")
        try:
            result = base64.b64decode(data, validate=True)
        except ValueError as e:
            raise ValueError("Invalid video base64") from e
    else:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Video requires HTTP(S) or base64 data:video; local files use --video")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(url, timeout=30) as source:
                result = bytearray()
                while True:
                    block = source.read(min(65536, MAX_BYTES+1-len(result)))
                    result.extend(block)
                    if len(result)>MAX_BYTES or time.monotonic()-started>30:
                        raise ValueError("Video download exceeds 20 MiB or 30 seconds")
                    if not block:
                        break
                result = bytes(result)
        except OSError as e:
            raise ValueError("Cannot download video") from e
    if not result or len(result)>MAX_BYTES:
        raise ValueError("Video must contain 1 byte..20 MiB")
    return result


def sample_video(data, fps, max_frames, resize):
    """Uniform targets across the clip; retain actual presentation timestamps.

    Only selected frames are converted to RGB. resize receives source H/W and
    target frame count, allowing the model's token budget to bound host RAM.
    """
    try:
        import av
        import numpy as np
    except ImportError as e:
        raise RuntimeError("Video input requires PyAV: python3 -m pip install 'av>=14,<17'") from e
    if not math.isfinite(fps) or not 0<fps<=30 or not 2<=max_frames<=64:
        raise ValueError("Video sampling requires fps in (0,30] and max_frames in 2..64")
    started = time.monotonic()
    def reject_external(url, flags, options):
        # A downloaded file must be self-contained. Do not follow playlist or
        # container references around the transport's byte/time boundaries.
        raise ValueError("Video containers may not open external resources")
    try:
        with av.open(io.BytesIO(data), io_open=reject_external, timeout=30) as container:
            if not container.streams.video:
                raise ValueError("File contains no video stream")
            stream = container.streams.video[0]
            stream.thread_type = "SLICE"
            stream.codec_context.thread_count = 1
            duration = float(stream.duration*stream.time_base) if stream.duration is not None else float(container.duration or 0)/av.time_base
            rate = float(stream.average_rate or 0)
            if not math.isfinite(duration) or not 0<duration<=MAX_SECONDS:
                raise ValueError("Video must have known duration of at most 120 seconds")
            count = min(max_frames, max(2, int(duration*fps)))
            if stream.frames>0:
                count = min(count, stream.frames)
            h, w = stream.codec_context.height, stream.codec_context.width
            if min(h,w)<=0 or h*w>32*1024*1024:
                raise ValueError("Video frames must be at most 32 megapixels")
            rh, rw = resize(h,w,count)
            targets = np.linspace(0, max(0,duration-(1/rate if rate>0 else duration/count)), count)
            frames, timestamps = [], []
            origin = float(stream.start_time*stream.time_base) if stream.start_time is not None else None
            previous = -1.0
            last = None
            for index, frame in enumerate(container.decode(stream)):
                if index>=36000 or time.monotonic()-started>30:
                    raise ValueError("Video decode exceeds frame/time limit")
                if frame.width!=w or frame.height!=h or frame.time is None:
                    raise ValueError("Video requires fixed dimensions and frame timestamps")
                if origin is None:
                    origin = float(frame.time)
                stamp = float(frame.time)-origin
                if not math.isfinite(stamp) or stamp<0 or stamp<previous or stamp>MAX_SECONDS:
                    raise ValueError("Invalid video presentation timestamps")
                previous = stamp
                last = frame
                if len(frames)<count and stamp+1e-6>=targets[len(frames)]:
                    # Pillow's bicubic image path is shared by native adapters.
                    from PIL import Image
                    picture = frame.to_image().resize((rw,rh), Image.Resampling.BICUBIC)
                    while len(frames)<count and stamp+1e-6>=targets[len(frames)]:
                        frames.append(picture)
                        timestamps.append(stamp)
                if len(frames)==count:
                    break
            if last is None:
                raise ValueError("Video contains no decodable frame")
            if len(frames)<count:
                from PIL import Image
                frames.append(last.to_image().resize((rw,rh), Image.Resampling.BICUBIC))
                timestamps.append(previous)
            return frames, timestamps, dict(duration=duration, source_fps=rate, sampled_frames=len(frames))
    except av.FFmpegError as e:
        raise ValueError("Invalid or unsupported video encoding") from e
