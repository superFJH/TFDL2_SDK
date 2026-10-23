"""Board warm ConvTranspose2d benchmark, fixed operands, no model downloads."""
import argparse,json,os,statistics,time
import torch
from torch.nn import functional as F
from tfllm.torch import NpuRuntime,NpuConvTranspose2d

def measure(fn,warmup,repeat):
    start=time.perf_counter();y=fn();cold=(time.perf_counter()-start)*1000
    for _ in range(warmup):y=fn()
    times=[]
    for _ in range(repeat):
        start=time.perf_counter();y=fn();times.append((time.perf_counter()-start)*1000)
    return y,{'cold_ms':cold,'median_ms':statistics.median(times),'samples_ms':times}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repeat',type=int,default=20);p.add_argument('--warmup',type=int,default=5)
    args=p.parse_args()
    if args.repeat<1 or args.warmup<0:p.error('repeat must be positive; warmup nonnegative')
    torch.set_num_threads(32);torch.manual_seed(219)
    print('DECONV_CONFIG '+json.dumps({'torch_threads':32,'environment':{k:os.environ.get(k) for k in ('TFLLM_NPU_MAX_CORES','TFLLM_CPU_THREADS','TFLLM_NPU_VERIFY_OUTPUTS')}}),flush=True)
    with torch.inference_mode(),NpuRuntime(memory_fraction=.125) as r:
        for name,ci,co,h,k,pad in [('sam_up1',256,64,64,2,0),('sam_up2',64,32,128,2,0),('overlap3',64,128,32,3,1)]:
            x=torch.randn(1,ci,h,h).half().float();w=(torch.randn(ci,co,k,k)*.03).half().float()
            want,cpu=measure(lambda:F.conv_transpose2d(x,w,stride=2,padding=pad),args.warmup,args.repeat)
            for dtype in (torch.float16,torch.float32):
                a=x.to(dtype);layer=NpuConvTranspose2d(w.to(dtype),padding=pad,runtime=r)
                y,timing=measure(lambda:layer(a),args.warmup,args.repeat)
                av,bv=y.double().flatten(),want.double().flatten()
                print('DECONV_RESULT '+json.dumps({'name':name,'input':list(a.shape),'weight':list(w.shape),'dtype':str(dtype),
                    'npu':timing,'torch_fp32':cpu,'cos':float(torch.dot(av,bv)/(av.norm()*bv.norm())),
                    'relative_l2':float((av-bv).norm()/bv.norm()),'output_dtype':str(y.dtype)}),flush=True)
                layer.close()

if __name__=='__main__':main()
