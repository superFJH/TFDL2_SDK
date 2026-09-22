"""Video model registry: transport/decode is shared, tensor/prompt rules are not."""
import base64
import hashlib
import math
import threading
from collections import OrderedDict

from .videos import read_video, sample_video


class Qwen3Video:
    @staticmethod
    def prepare(data, config):
        import numpy as np
        factor = config["patch"]*config["merge"]
        budget = config["max_video_tokens"]

        def resize(h,w,count):
            if max(h,w)/min(h,w)>200:
                raise ValueError("Video aspect ratio exceeds 200")
            pairs = (count+1)//2
            per_pair = min(config["max_image_tokens"], budget//pairs)
            if per_pair<1:
                raise ValueError("Video token budget too small for the sampled frames")
            rh,rw = max(factor,round(h/factor)*factor),max(factor,round(w/factor)*factor)
            if rh*rw>per_pair*factor*factor:
                scale = math.sqrt(h*w/(per_pair*factor*factor))
                rh,rw = max(factor,math.floor(h/scale/factor)*factor),max(factor,math.floor(w/scale/factor)*factor)
                if rh*rw>per_pair*factor*factor:
                    if rw>rh: rw=factor*max(1,per_pair//(rh//factor))
                    else: rh=factor*max(1,per_pair//(rw//factor))
            return rh,rw

        frames, stamps, metadata = sample_video(data, config["fps"], config["max_frames"], resize)
        if len(frames)%2:
            frames.append(frames[-1]); stamps.append(stamps[-1])
        spans = []
        mean = np.asarray(config["mean"],dtype=np.float32)
        std = np.asarray(config["std"],dtype=np.float32)
        for i in range(0,len(frames),2):
            # Real C,T,H,W patch extraction happens in C++; storage is TCHW.
            pixels = np.stack([np.asarray(f,dtype=np.float32) for f in frames[i:i+2]])/255.0
            pixels = ((pixels-mean)/std).transpose(0,3,1,2)
            payload = np.ascontiguousarray(pixels,dtype="<f2").tobytes()
            h,w = pixels.shape[-2:]
            spans.append(dict(kind="video",frames=2,height=h,width=w,timestamp=(stamps[i]+stamps[i+1])/2,
                              pixels=base64.b64encode(payload).decode("ascii")))
        metadata["visual_tokens"] = sum(s["height"]*s["width"]//factor**2 for s in spans)
        if metadata["visual_tokens"]>budget:
            raise ValueError("Video exceeds total visual token budget")
        return spans, metadata


ADAPTERS = {"qwen3vl_merger": Qwen3Video}


class VideoPreprocessor:
    def __init__(self):
        self.cache = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()

    def prepare(self, part, config):
        adapter = ADAPTERS.get(config.get("projector"))
        if adapter is None:
            raise ValueError("This model has no native video adapter; currently Qwen3-VL is supported")
        data = read_video(part)
        key = (hashlib.sha256(data).digest(), repr(sorted(config.items())))
        with self.lock:
            if key in self.cache:
                result,size = self.cache.pop(key);self.cache[key]=(result,size)
                return result
        result = adapter.prepare(data,config)
        size = sum(len(s["pixels"]) for s in result[0])
        with self.lock:
            if key not in self.cache and size<=64*1024*1024:
                while self.cache and (len(self.cache)>=8 or self.bytes+size>64*1024*1024):
                    _,(_,old)=self.cache.popitem(last=False);self.bytes-=old
                self.cache[key]=(result,size);self.bytes+=size
        return result


def qwen_video_text(spans):
    return "".join(f'<{s["timestamp"]:.1f} seconds><|vision_start|><|video_pad|><|vision_end|>' for s in spans)
