"""Opt-in full official model oracle; local weights or deterministic random models."""
import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from tfllm.torch import NpuRuntime
from tfllm.vision import attach
from tfllm.vision.loaders import build


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family",required=True,choices=("efficienttam","depth-anything-v2","depth-anything-v3"))
    p.add_argument("--checkpoint")
    p.add_argument("--config",default="configs/efficienttam/efficienttam_ti_512x512.yaml")
    p.add_argument("--encoder",default="vits")
    p.add_argument("--model-name",default="da3-small")
    p.add_argument("--backend",choices=("cpu","npu"),default="cpu")
    p.add_argument("--input-size",type=int,default=56)
    args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(191)
    if args.checkpoint:
        model,processor,provider=build(args.family,args.checkpoint,provider="official",
                                      config=args.config,encoder=args.encoder)
    elif args.family=="efficienttam":
        from efficient_track_anything.build_efficienttam import build_efficienttam
        from efficient_track_anything.efficienttam_image_predictor import EfficientTAMImagePredictor
        model=build_efficienttam(args.config,device="cpu",hydra_overrides_extra=["++model.compile_image_encoder=False"]).eval()
        processor=EfficientTAMImagePredictor(model);provider="official"
    elif args.family=="depth-anything-v2":
        from depth_anything_v2.dpt import DepthAnythingV2
        model=DepthAnythingV2(encoder="vits",features=64,out_channels=[48,96,192,384]).eval()
        processor=model;provider="official"
    else:
        from depth_anything_3.api import DepthAnything3
        model=DepthAnything3(args.model_name).cpu().eval();processor=model;provider="official"
    image=np.arange(48*64*3,dtype=np.int64).reshape(48,64,3).astype(np.uint8)
    images=[image,np.flip(image,axis=1).copy()]
    points=np.array([[24.,20.]]);labels=np.array([1])
    with torch.inference_mode():
        if args.family=="efficienttam":
            processor.set_image(image)
            _,scores,logits=processor.predict(point_coords=points,point_labels=labels)
            gold={"scores":scores,"low_res_masks":logits}
        elif args.family=="depth-anything-v2":
            pixels,_=model.image2tensor(image[...,::-1].copy(),args.input_size)
            depth=model(pixels.cpu())
            gold={"depth":F.interpolate(depth[:,None],image.shape[:2],mode="bilinear",align_corners=True)[0,0].numpy()}
        else:
            result=model.inference(images,process_res=args.input_size)
            gold={k:getattr(result,k) for k in ("depth","conf","extrinsics","intrinsics")}
    with NpuRuntime(backend=args.backend) as runtime,attach(model,family=args.family,runtime=runtime,
                                                          processor=processor,provider=provider) as session:
        if args.family=="efficienttam":
            session.set_image(image)
            _,scores,logits=session.predict(point_coords=points,point_labels=labels)
            actual={"scores":scores,"low_res_masks":logits}
        elif args.family=="depth-anything-v2":
            actual={"depth":session.infer_image(image[...,::-1].copy(),input_size=args.input_size)}
        else:
            result=session.inference(images,process_res=args.input_size)
            actual={k:getattr(result,k) for k in gold}
        errors={}
        for k,g in gold.items():
            y=actual[k]
            assert y.shape==g.shape and np.isfinite(y).all(),k
            delta=y.astype(np.float64)-g.astype(np.float64)
            relative=float(np.sqrt(np.mean(delta**2))/(np.sqrt(np.mean(g.astype(np.float64)**2))+1e-6))
            errors[k]={"max_abs":float(np.max(np.abs(delta))),"relative_rmse":relative}
            assert relative<(.1 if args.backend=="cpu" else .25),(k,errors[k])
        report=session.report.to_dict()
        assert any(c["status"] in ("npu","cpu_reference") for c in report["calls"])
        print("VISION_OFFICIAL "+json.dumps({"family":args.family,"backend":args.backend,
            "checkpoint":bool(args.checkpoint),"task_quality_tested":False,"errors":errors,"coverage":report}))


if __name__=="__main__":
    main()
