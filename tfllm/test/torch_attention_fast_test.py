"""General FAST attention: public switches, layouts, sparse/masked numerical oracles."""
import hashlib
import os
import unittest
from unittest import mock
import torch
from torch.nn import functional as F
from tfllm.torch import NpuRuntime, optimize, restore, Unsupported
from tfllm.vision import attach

torch.set_num_threads(4)
BACKEND=os.environ.get('TFLLM_TEST_BACKEND','npu')
FRACTION=float(os.environ.get('TFLLM_TEST_MEMORY_FRACTION','.8'))

class GeneralFast(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(320)
        self.rt=NpuRuntime(backend=BACKEND,memory_fraction=FRACTION,attention_fast=True,
                           attention_policy='hybrid',attention_fuse_workers=16,attention_score_mib=4)
    def tearDown(self):self.rt.close()
    def close_to(self,y,gold,label):
        self.assertTrue(torch.isfinite(y).all(),label)
        error=(y.float()-gold.float());l2=float(error.norm()/gold.float().norm().clamp_min(1e-8))
        maximum=float(error.abs().max())
        print('FAST_ORACLE',label,'l2',l2,'max_abs',maximum,flush=True)
        self.assertLess(l2,.10,label)
        torch.testing.assert_close(y,gold,atol=.10,rtol=.10)
    def test_dense_cross_gqa_tails(self):
        cases=[(2,3,3,17,97,32,48,False),(1,6,2,33,67,96,72,True),
               (1,2,2,1,129,36,20,False),(1,2,2,19,73,192,160,False),
               (1,1,1,9,17,256,256,True)]
        with torch.inference_mode():
            for b,h,kh,n,k,d,dv,causal in cases:
                for dtype in [torch.float32,torch.float16]:
                    q=torch.randn(b,h,n,d,dtype=dtype);key=torch.randn(b,kh,k,d,dtype=dtype);v=torch.randn(b,kh,k,dv,dtype=dtype)
                    kw=dict(is_causal=causal,enable_gqa=h!=kh,scale=.7/d**.5)
                    y=self.rt.attention(q,key,v,**kw);gold=F.scaled_dot_product_attention(q,key,v,**kw)
                    self.close_to(y,gold,str((b,h,kh,n,k,d,dv,causal,dtype)))
    def test_strided_and_window(self):
        with torch.inference_mode():
            for dtype in [torch.float32,torch.float16]:
                q,k,v=[torch.randn(2,64,4,32,dtype=dtype).transpose(1,2) for _ in range(3)]
                self.close_to(self.rt.attention(q,k,v),F.scaled_dot_product_attention(q,k,v),'strided '+str(dtype))
                q,k,v=[torch.randn(4,8,64,32,dtype=dtype) for _ in range(3)]
                self.close_to(self.rt.attention(q,k,v),F.scaled_dot_product_attention(q,k,v),'window '+str(dtype))
    def test_intervals_and_blocks(self):
        with torch.inference_mode():
            for dtype in [torch.float32,torch.float16]:
                q=torch.randn(1,2,17,64,dtype=dtype);k=torch.randn(1,2,37,64,dtype=dtype);v=torch.randn(1,2,37,48,dtype=dtype)
                bounds=torch.tensor([[[r%5,30-r%3] for r in range(17)]],dtype=torch.int64);bounds[:,3]=0
                cols=torch.arange(37);mask=((cols>=bounds[...,0,None])&(cols<bounds[...,1,None]))[:,None]
                y=self.rt.attention_intervals(q,k,v,bounds)
                self.close_to(y,F.scaled_dot_product_attention(q,k,v,attn_mask=mask),'intervals '+str(dtype))
                self.assertTrue((y[:,:,3]==0).all())
                offsets=torch.tensor([0,2,2,3,4,6,7],dtype=torch.int64);indices=torch.tensor([0,2,1,2,0,1,0],dtype=torch.int64)
                dense=torch.zeros(1,2,17,37,dtype=torch.bool)
                for head in range(2):
                    for block in range(3):
                        row=head*3+block
                        for col in indices[offsets[row]:offsets[row+1]]:
                            dense[0,head,block*8:min(17,(block+1)*8),int(col)*16:min(37,(int(col)+1)*16)]=True
                y=self.rt.attention_blocks(q,k,v,offsets,indices,query_block_size=8,kv_block_size=16)
                self.close_to(y,F.scaled_dot_product_attention(q,k,v,attn_mask=dense),'blocks '+str(dtype))
    def test_arbitrary_head_masks_and_bias(self):
        with torch.inference_mode():
            for dtype in [torch.float32,torch.float16]:
                q=torch.randn(2,3,17,32,dtype=dtype);k=torch.randn(2,3,37,32,dtype=dtype);v=torch.randn(2,3,37,48,dtype=dtype)
                visible=torch.rand(2,3,17,37)>.3;visible[:,:,4]=False
                bias=torch.randn(2,3,17,37)*.2;bias.masked_fill_(~visible,-torch.inf)
                for mask in [visible,bias]:
                    for causal in [False,True]:
                        y=self.rt.attention(q,k,v,attn_mask=mask,is_causal=causal)
                        refmask=mask
                        if causal:
                            lower=torch.ones(17,37,dtype=torch.bool).tril()
                            refmask=mask & lower if mask.dtype==torch.bool else mask.masked_fill(~lower,-torch.inf)
                        self.close_to(y,F.scaled_dot_product_attention(q,k,v,attn_mask=refmask),'arbitrary '+str((dtype,mask.dtype,causal)))
                        self.assertTrue((y[:,:,4]==0).all())

    def test_routes_and_unsupported_inputs(self):
        class Model(torch.nn.Module):
            def forward(self,q,bias=None):return F.scaled_dot_product_attention(q,q,q,attn_mask=bias)
        model=Model().eval();q=torch.randn(1,2,17,32)
        with attach(model,family='sam2',runtime=self.rt) as session:
            self.close_to(session(q),F.scaled_dot_product_attention(q,q,q),'hybrid override')
            self.assertTrue(any(c['status']=='npu_fast' for c in session.report.to_dict()['calls']))
            bias=torch.randn(17,17)*.2
            self.close_to(session(q,bias),F.scaled_dot_product_attention(q,q,q,attn_mask=bias),'finite bias')
            with self.assertRaisesRegex(ValueError,'finite'):
                session(q,torch.full((17,17),float('nan')))
        with torch.inference_mode():
            q=torch.randn(1,1,3,257)
            with self.assertRaisesRegex((ValueError,RuntimeError),'256'):self.rt.attention(q,q,q)
    def test_packed_mha_scopes(self):
        class Packed(torch.nn.Module):
            def __init__(self):super().__init__();self.mha=torch.nn.MultiheadAttention(32,4,batch_first=True)
            def forward(self,x):return self.mha(x,x,x,need_weights=False)[0]
        m=Packed().eval();x=torch.randn(1,17,32)
        with torch.inference_mode():gold=m(x)
        with attach(m,family='sam3',runtime=self.rt) as session:
            self.close_to(session(x),gold,'packed Torch MHA')
            self.assertTrue(any(c['status']=='npu_fast' for c in session.report.to_dict()['calls']))

    def test_eager_optimize_functional_scope_and_restore(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.mha=torch.nn.MultiheadAttention(32,4,batch_first=True)
            def forward(self,x):return self.mha(x,x,x,need_weights=False)[0]
        m=Model().eval();x=torch.randn(1,17,32)
        with torch.inference_mode():gold=m(x)
        optimize(m,runtime=self.rt)
        try:
            with torch.inference_mode():actual=m(x)
            self.close_to(actual,gold,'eager optimize')
            self.assertTrue(any(c['status']=='npu_fast' for c in m._tfllm_optimization.report.to_dict()['calls']))
        finally:restore(m)
        self.assertNotIn('forward',m.__dict__)
        with torch.inference_mode():torch.testing.assert_close(m(x),gold,atol=0,rtol=0)

    def test_sam3_decoder_unused_weights_scope(self):
        class TransformerDecoderLayer(torch.nn.Module):
            def __init__(self):super().__init__();self.mha=torch.nn.MultiheadAttention(32,4,batch_first=True)
            def forward(self,x):return self.mha(x,x,x)[0]
        # Exercise the audited adapter without requiring the optional provider.
        TransformerDecoderLayer.__module__='sam3.model.decoder'
        m=TransformerDecoderLayer().eval();x=torch.randn(1,17,32)
        with torch.inference_mode():gold=m(x)
        optimize(m,runtime=self.rt,operators=('attention',))
        try:
            with torch.inference_mode():actual=m(x)
            self.close_to(actual,gold,'SAM3 unused weights')
        finally:restore(m)
        self.assertNotIn('forward',m.__dict__)
        # Generic callers still receive a clear error for requested weights.
        class Packed(torch.nn.Module):
            def forward(self,x):return m.mha(x,x,x)
        p=Packed().eval();optimize(p,runtime=self.rt,operators=('attention',))
        try:
            with torch.inference_mode(),self.assertRaisesRegex(Unsupported,'need_weights=False'):p(x)
        finally:restore(p)

    def test_environment_and_explicit_override(self):
        with mock.patch.dict(os.environ,{'TFLLM_NPU_ATTENTION_FAST':'1'}):
            with NpuRuntime(backend=BACKEND,memory_fraction=FRACTION) as enabled:self.assertTrue(enabled.attention_fast)
            with NpuRuntime(backend=BACKEND,memory_fraction=FRACTION,attention_fast=False) as disabled:
                self.assertFalse(disabled.attention_fast)
                q=torch.randn(1,1,9,32)
                with torch.inference_mode():self.assertTrue(torch.isfinite(disabled.attention(q,q,q)).all())
        with self.assertRaises(TypeError):NpuRuntime(backend=BACKEND,attention_fast=1)
        with self.assertRaises(ValueError):NpuRuntime(backend='cpu',attention_fast=True)

if __name__=='__main__':unittest.main()
