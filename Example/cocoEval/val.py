import sys
import json
import argparse
import torch.utils.cpp_extension
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.trackers import BOTSORT,BYTETracker
from ultralytics.utils import ops
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from TFDL2 import TFContext,TFExecutor,TFDataType,Option
import yaml 
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results,Boxes
from types import SimpleNamespace
import numpy as np
import torch
import cv2
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from pathlib import Path
from ultralytics.utils.ops import Profile
import argparse

def pre_transform(im,imgsz):
        same_shapes = all(x.shape == im[0].shape for x in im)
        letterbox = LetterBox(imgsz, auto=same_shapes, stride=32)
        return [letterbox(image=x) for x in im]

def preprocess(im,imgsz):
        im = np.stack(pre_transform(im,imgsz))
        #im = cv2.resize(im[0],imgsz)
        #im = im[np.newaxis,:]
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        #im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

class TFDLValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
         
         super().__init__(dataloader, save_dir, args, _callbacks)
    
    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].to(torch.uint8)#(batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)
        '''
        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]
        '''
        return batch

    def __call__(self, trainer=None, model=None):
        self.training = False
        self.stride = 32
        if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
            self.data = check_det_dataset(self.args.data)
        elif self.args.task == "classify":
            self.data = check_cls_dataset(self.args.data, split=self.args.split)
        else:
            raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))
        self.dataloader = self.get_dataloader(self.data.get(self.args.split), 1)
        dt = (
            Profile(device=torch.device("cpu")),
            Profile(device=torch.device("cpu")),
            Profile(device=torch.device("cpu")),
            Profile(device=torch.device("cpu")),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(model)
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            #preds = model(batch["img"], augment=False)
            with dt[1]:
                preds = self.runModel(batch["img"])
            #print(batch["img"].shape)

            with dt[2]:
                pass

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        self.gather_stats()
        stats = self.get_stats()
        #self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
        #if self.args.save_json and self.jdict:
        with open(str(self.save_dir / "predictions.json"), "w") as f:
            LOGGER.info(f"Saving {f.name}...")
            json.dump(self.jdict, f)  # flatten and save
        stats = self.eval_json(stats)  # update stats
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        return stats
            
    
    def prepareModel(self,modelpath,core=[1],imgsz=(640,640)):
        context = TFContext(path=modelpath)
        newopention = Option
        newopention["Core"] = core
        newopention["ignoreDepthwise"] = True
        newopention["optimize"]["MakeAlign"] = True
        newopention["optimize"]["AttnSoftmaxImpl"] = True
        newopention["InputShape"] = [
            {"NodeName":"TFDL_Placeholder_0","Shape":[1,3,imgsz[0],imgsz[1]]},
        ]
        self.option = newopention
        self.context = context
        key = f'{imgsz[0]}_{imgsz[1]}'
        self.model = {key:TFExecutor(context=context,config=newopention)}
        
        

    def runModel(self,img):

        if self.option["InputShape"][0]["Shape"] != [1,3,*img.shape[2:]] and f'{img.shape[2]}_{img.shape[3]}' not in self.model.keys():
            option = self.option
            option["InputShape"][0]["Shape"] = [1,3,*img.shape[2:]]
            print()
            print("新增尺寸:",option["InputShape"][0]["Shape"])
            self.model[f'{img.shape[2]}_{img.shape[3]}'] = TFExecutor(context=self.context,config=option)
            model = self.model[f'{img.shape[2]}_{img.shape[3]}']
        else:
            model = self.model[f'{img.shape[2]}_{img.shape[3]}']
        img = img.numpy()
        inputs = model.GetInputs()[0]
        inputs.fromNumpy(img)
        out = model()[0].toNumpy()                    
        out = torch.from_numpy(out)
        torch.utils.cpp_extension
        return out
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 TFDL 进行coco eval模型精度验证")
    parser.add_argument("--model-path", type=str, required=True, help="TFDL模型路径")
    parser.add_argument("--model-name", type=str, required=True, help="最终测试模型名")
    parser.add_argument("--save-dir", type=str,  help="保存路径", default="runs/val")
    parser.add_argument("--config", type=str, help="coco.yaml路径", default="coco.yaml")
    parser.add_argument("--pt-path", type=str, required=True, help="pt模型路径,用来init_metrics，因为很多模型的metrics不一样")
    args = parser.parse_args()

    modelname = args.model_name
    
    save_dir = Path(args.save_dir) / modelname

    validargs = dict(mode="val",rect=True,data=args.config,batch=1,save_json=True)

    validator = TFDLValidator(args=validargs,save_dir=save_dir)
    validator.prepareModel(args.model_path)
    metrics = validator(model=YOLO(args.pt_path))





