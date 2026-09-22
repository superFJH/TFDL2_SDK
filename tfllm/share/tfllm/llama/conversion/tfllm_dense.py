"""Pinned HF dense vision import additions. No remote Python model code."""
from .base import MmprojModel, ModelBase, gguf
from .qwen import Qwen2Model

@ModelBase.register("TFLLMLocateText")
class LocateText(Qwen2Model):
    model_arch = gguf.MODEL_ARCH.QWEN2

    def set_vocab(self):
        super().set_vocab()
        # The released tokenizer_config still contains Qwen's text template;
        # the multimodal template lives in chat_template.json. Carry it into
        # standalone and paired decode GGUFs as well as the CLI manifest.
        import json
        path = self.dir_model / "chat_template.jinja"
        if path.is_file():
            template = path.read_text()
        elif (self.dir_model / "chat_template.json").is_file():
            template = json.loads((self.dir_model / "chat_template.json").read_text())["chat_template"]
        else:
            return
        self.gguf_writer.remove_key("tokenizer.chat_template")
        self.gguf_writer.add_chat_template(template)

    def set_gguf_parameters(self):
        if self.hparams.get("model_type") != "qwen2":
            raise ValueError("LocateAnything slow mode currently requires Qwen2")
        super().set_gguf_parameters()
        self.gguf_writer.add_string("tfllm.source_architecture", "locateanything")
        self.gguf_writer.add_string("tfllm.generation_mode", "slow")

    @classmethod
    def filter_tensors(cls, item):
        name, _ = item
        if name.startswith(("vision_model.", "mlp1.")):
            return None
        if not name.startswith("language_model."):
            raise ValueError("Unexpected LocateAnything text tensor: " + name)
        return super().filter_tensors(item)

@ModelBase.register("TFLLMLocateVision")
class LocateVision(MmprojModel):
    """LocateAnything's MoonViT-SO, not Kimi-VL's projector/RoPE/interpolation.

    Reference: nvidia/LocateAnything-3B modeling_vit.py and
    modeling_locateanything.py. Export only; no remote code is executed.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c = self.hparams_vision
        if c.get("model_type") != "moonvit" or c.get("merge_kernel_size") != [2, 2]:
            raise ValueError("LocateAnything requires 2D MoonViT with 2x2 merge")
        if c["init_pos_emb_height"] != c["init_pos_emb_width"]:
            raise ValueError("LocateAnything learned position grid must be square")
        c["image_size"] = c["init_pos_emb_height"] * c["patch_size"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type("locateanything")
        self.gguf_writer.add_vision_use_gelu(True)
        self.gguf_writer.add_vision_projector_scale_factor(2)
        self.gguf_writer.add_vision_attention_layernorm_eps(1e-5)

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if name.startswith("language_model."):
            return None
        if not name.startswith(("vision_model.", "mlp1.")):
            raise ValueError("Unexpected LocateAnything tensor: " + name)
        return name, gen

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if new_name == "v.position_embd.weight":
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, value, name, bid):
        import re
        n = name.removeprefix("vision_model.")
        for old, new in (("patch_embed.proj", "v.patch_embd"),
                         ("patch_embed.pos_emb", "v.position_embd"),
                         ("encoder.final_layernorm", "v.post_ln"),
                         ("mlp1.0", "mm.input_norm"), ("mlp1.1", "mm.1"), ("mlp1.3", "mm.2")):
            n = n.replace(old, new)
        n = re.sub(r"encoder.blocks\.(\d+)\.", r"v.blk.\1.", n)
        for old, new in (("norm0", "ln1"), ("norm1", "ln2"), ("wqkv", "attn_qkv"),
                         ("wo", "attn_out"), ("mlp.fc0", "ffn_up"), ("mlp.fc1", "ffn_down")):
            n = n.replace(old, new)
        if not n.startswith(("v.", "mm.")):
            raise ValueError("Unsupported LocateAnything vision tensor: " + name)
        if n == "v.position_embd.weight":
            value = value.reshape(-1, self.hparams["hidden_size"])
        yield n, value

@ModelBase.register("TFLLMDenseVision")
class DenseVision(MmprojModel):
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        c=self.hparams
        self.gguf_writer.add_clip_projector_type("mlp")
        self.gguf_writer.add_vision_attention_layernorm_eps(c.get("layer_norm_eps",1e-5))
        act=c.get("hidden_act","quick_gelu")
        if act in ("gelu","gelu_pytorch_tanh"): self.gguf_writer.add_vision_use_gelu(True)
        elif act!="quick_gelu": raise ValueError("Unsupported CLIP/SigLIP activation: "+act)
        strategy=self.global_config.get("vision_feature_select_strategy","default")
        expected="full" if c.get("model_type")=="siglip_vision_model" else "default"
        if strategy!=expected:
            raise ValueError("CLIP/SigLIP MLP currently requires patch-only feature selection")
        selected=self.global_config.get("vision_feature_layer",-2)
        layers=[selected] if isinstance(selected,int) else selected
        depth=c["num_hidden_layers"]
        layers=[i+depth+1 if i<0 else i for i in layers]
        if not layers or len(layers)!=len(set(layers)) or any(i<0 or i>depth for i in layers):raise ValueError("Invalid vision feature layers")
        self.gguf_writer.add_array("clip.vision.feature_layer",layers)

    @classmethod
    def filter_tensors(cls,item):
        name,gen=item
        if name.startswith("model."):name=name[6:]
        if name=='image_newline':
            raise ValueError("LLaVA-NeXT spatial-unpad/image-newline assembly is not supported; refusing to drop its learned row separators")
        if not name.startswith(("vision_tower.","multi_modal_projector.")):return None
        # CLIP post norm is applied only to pooled CLS, which LLaVA discards.
        # SigLIP's post norm belongs to last_hidden_state, not hidden_states.
        if '.post_layernorm.' in name or '.head.' in name:return None
        return name,gen

    def tensor_force_quant(self,name,new_name,bid,n_dims):
        if 'position_embd' in new_name:return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name,new_name,bid,n_dims)

    def modify_tensors(self,data_torch,name,bid):
        for source,target in (("multi_modal_projector.linear_1.","mm.0."),("multi_modal_projector.linear_2.","mm.2.")):
            if name.startswith(source):
                yield target+name[len(source):],data_torch
                return
        yield from super().modify_tensors(data_torch,name,bid)

from .internvl import InternVisionModel
import re
@ModelBase.register("TFLLMInternVision")
class DenseInternVision(InternVisionModel):
    @classmethod
    def filter_tensors(cls,item):
        name,gen=item
        if name.startswith('model.'):name=name[6:]
        if not name.startswith(('vision_tower.','multi_modal_projector.')):return None
        return name,gen

    def modify_tensors(self,value,name,bid):
        n=name.removeprefix('vision_tower.')
        n=n.replace('embeddings.cls_token','v.class_embd').replace('embeddings.position_embeddings','v.position_embd.weight')
        n=n.replace('embeddings.patch_embeddings.projection.','v.patch_embd.')
        n=re.sub(r'encoder.layer\.(\d+)\.',r'v.blk.\1.',n)
        for old,new in [('attention.q_proj','attn_q'),('attention.k_proj','attn_k'),('attention.v_proj','attn_v'),('attention.projection_layer','attn_out'),('attention.q_norm','attn_q_norm'),('attention.k_norm','attn_k_norm'),('mlp.fc1','ffn_up'),('mlp.fc2','ffn_down'),('layernorm_before','ln1'),('layernorm_after','ln2'),('lambda_1','ls1.weight'),('lambda_2','ls2.weight'),('multi_modal_projector.layer_norm','mm.0'),('multi_modal_projector.linear_1','mm.1'),('multi_modal_projector.linear_2','mm.3')]:n=n.replace(old,new)
        if n=='v.class_embd':value=value.reshape(-1)
        if n=='v.position_embd.weight':value=value.reshape(-1,self.hparams['hidden_size'])
        if not n.startswith(('v.','mm.')):raise ValueError('Unsupported InternVL vision tensor: '+name)
        if n.startswith('mm.'):n='mm.model.mlp.'+n[3:]
        yield n,value
