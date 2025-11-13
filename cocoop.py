import torch
import torch.nn as nn
from transformers import AutoModel, CLIPProcessor, AutoTokenizer
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)
from transformers.modeling_outputs import BaseModelOutput
from torch.nn import functional as F

# 加载 CLIP
model_name = "openai/clip-vit-base-patch16"
clip = AutoModel.from_pretrained(model_name).eval()

vision_dim = clip.visual_projection.in_features
text_dim = clip.text_projection.in_features


# MetaNet
class MetaNet(nn.Module):
    def __init__(self, vision_dim, text_dim):
        super().__init__()
        hidden = vision_dim // 16
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, hidden), nn.ReLU(), nn.Linear(hidden, text_dim)
        )

    def forward(self, img_feat):
        return self.mlp(img_feat)  # (B, text_dim)


# CS-Prompt Learner with dynamic prompt + correct embeddings & pos encoding
class ConditionalPromptLearner(nn.Module):
    def __init__(self, clip_name, classnames, n_ctx=4):
        super().__init__()
        self.clip = AutoModel.from_pretrained(clip_name).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(clip_name)
        self.n_ctx = n_ctx
        self.vision_dim = self.clip.visual_projection.in_features
        self.text_dim = self.clip.text_projection.in_features
        self.ctx = nn.Parameter(torch.randn(1, n_ctx, self.text_dim) * 0.02)

        self.token_embedding = self.clip.text_model.embeddings.token_embedding  # token embedding
        self.embedding = self.clip.text_model.embeddings
        self.text_encoder = self.clip.text_model.encoder
        self.final_layer_norm = self.clip.text_model.final_layer_norm

        self.metanet = MetaNet(self.vision_dim, self.text_dim)
        
        logit_scale_init = 1 / 0.07  # CLIP 使用的初始 logit scale
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)
        
        for param in self.clip.parameters():
            param.requires_grad = False


        self.class_embeds = self._init_class_embeds(classnames)

        self.register_buffer(
            "bos_embed",
            self.token_embedding.weight[
                self.tokenizer.bos_token_id
            ].clone().to(self.ctx.device),
            persistent=False,
        )  # (text_dim,)
        self.register_buffer(
            "eos_embed",
            self.token_embedding.weight[
                self.tokenizer.eos_token_id
            ].clone().to(self.ctx.device),
            persistent=False,
        )

        self._init_prompt()

    def _init_prompt(self):
        text = "a photo of a"
        text_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.n_ctx,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        with torch.no_grad():
            prompt_embeds = self.token_embedding(
                text_ids.to(self.ctx.device)
            )  # (1,n_ctx,text_dim)
        self.ctx.data.copy_(prompt_embeds)

    def _init_class_embeds(self, classnames):
        class_ids = self.tokenizer(
            classnames,
            add_special_tokens=False,
            max_length=16,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        with torch.no_grad():
            class_embeds = self.token_embedding(
                class_ids.to(self.ctx.device)
            )  # (n_class,16,text_dim)
        return class_embeds  # (n_class,16,text_dim)

    def encoder_forward(self, prompt):
        """
        prompt: (B,n_class,n_ctx+16+2,text_dim)
        clip text model从token embedding之后的处理过程
        """
        prompt_viewed = prompt.view(
            prompt.shape[0]*prompt.shape[1], prompt.shape[2], -1
        )  # (B*n_class,n_ctx+16+2,text_dim)

        # if prompt_viewed.size(1)<self.clip.text_model.config.max_position_embeddings:
        #     pad_len=self.clip.text_model.config.max_position_embeddings-prompt_viewed.size(1)

        inputs_embeds = prompt_viewed
        embeddings = self.embedding(inputs_embeds=inputs_embeds)

        attention_mask = torch.ones(
            inputs_embeds.size()[:-1], dtype=torch.long, device=inputs_embeds.device
        )
        attention_mask = _prepare_4d_attention_mask(attention_mask, embeddings.dtype)
        causal_attention_mask = _create_4d_causal_attention_mask(
            embeddings.shape[:2],
            dtype=embeddings.dtype,
            device=embeddings.device,
        )

        encoder_outputs: BaseModelOutput = self.text_encoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )

        last_hidden_state = (
            encoder_outputs.last_hidden_state
        )  # (B*n_class,n_ctx+16+2,text_dim)
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        
        pooled_output = last_hidden_state[
            :, -1, :
        ] # (B*n_class,text_dim) 取eos位置的输出作为句子表示
        
        return pooled_output

    def forward(self, pixel_values, labels=None):
        image_feats = self.clip.vision_model(
            pixel_values=pixel_values
        ).pooler_output  # (B,vision_dim)
        b = image_feats.shape[0]
        n_class = self.class_embeds.shape[0]
        pi = self.metanet(image_feats).unsqueeze(1)  # (B,1,text_dim)
        ctx = self.ctx.expand(b, -1, -1)  # (B,n_ctx,text_dim)
        ctx_pi = ctx + pi  # (B,n_ctx,text_dim) 广播
        ctx_pi = ctx_pi.unsqueeze(1).expand(b, n_class, -1, -1)  # (B,n_class,n_ctx,text_dim)
        prompt = torch.cat(
            [ctx_pi, self.class_embeds.unsqueeze(0).expand(b, -1, -1, -1)], dim=2
        )  # (B,n_class,n_ctx+16,text_dim)

        bos = (
            self.bos_embed.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(b, n_class, -1, -1)
        )  # (B,n_class,1,text_dim)
        eos = (
            self.eos_embed.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(b, n_class, -1, -1)
        )  # (B,n_class,1,text_dim)
        prompt = torch.cat([bos, prompt, eos], dim=2)  # (B,n_class,n_ctx+16+2,text_dim)

        text_feats = self.encoder_forward(prompt)  # (B*n_class,text_dim)
        text_feats=self.clip.text_projection(text_feats)  # (B*n_class,text_dim)
        text_feats = text_feats.view(
            b, -1, self.text_dim
        )  # (B,n_class,text_dim)
        
        image_feats=self.clip.visual_projection(image_feats)  # (B,text_dim)
        image_feats=image_feats.unsqueeze(1)  # (B,1,text_dim)
        
        logits=F.normalize(image_feats,dim=-1) @ F.normalize(text_feats,dim=-1).permute(0,2,1)  # (B,1,n_class)
        logits=self.logit_scale.exp()*logits  # scaled logits
        logits=logits.squeeze(1)  # (B,n_class)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            return logits
        
        
# 测试
if __name__ == "__main__":
    classnames = ["cat", "dog", "car", "bus"]
    model = ConditionalPromptLearner(model_name, classnames, n_ctx=4)
    processor = CLIPProcessor.from_pretrained(model_name)
    dummy_image = torch.randn(1, 3, 224, 224)
    logits = model(dummy_image)  # (1,n_class)
    print(logits)