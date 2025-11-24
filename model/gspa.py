import torch
import torch.nn as nn
from transformers import AutoModel, CLIPProcessor, AutoTokenizer
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)
from transformers.modeling_outputs import BaseModelOutput
from torch.nn import functional as F
from .modules.grl import grad_reverse
from .modules.output import GSPAOutput

# 加载 CLIP
model_name = "openai/clip-vit-base-patch16"


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
    def __init__(self, config):
        super().__init__()
        self.config = config
        clip_name = config.model_name_or_path
        classnames = config.class_names
        n_ctx = config.ctx

        self.clip = AutoModel.from_pretrained(clip_name)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_name)
        self.n_ctx = n_ctx
        self.vision_dim = self.clip.visual_projection.in_features
        self.text_dim = self.clip.text_projection.in_features
        self.ctx = nn.Parameter(torch.randn(1, n_ctx, self.text_dim) * 0.02)
        self.classnames = classnames

        # ---Text Model拆分---
        self.token_embedding = (
            self.clip.text_model.embeddings.token_embedding
        )  # token embedding
        # 不保存 embedding 的引用，避免与 token_embedding 重复
        # self.embedding = self.clip.text_model.embeddings  # 移除此行
        self.position_embedding = self.clip.text_model.embeddings.position_embedding
        self.text_encoder = self.clip.text_model.encoder
        self.final_layer_norm = self.clip.text_model.final_layer_norm

        # ---Vision Model拆分---
        self.vision_embedding = self.clip.vision_model.embeddings
        self.vision_pre_layernorm = self.clip.vision_model.pre_layrnorm
        self.vision_encoder = self.clip.vision_model.encoder
        self.vision_post_layernorm = self.clip.vision_model.post_layernorm

        self.metanet = MetaNet(self.vision_dim, self.text_dim)

        self.gate = nn.Sequential(
            nn.Linear(self.vision_dim, self.vision_dim // 16),
            nn.SiLU(),
            nn.Linear(self.vision_dim // 16, 1),
            nn.Sigmoid(),
        )

        self.logit_scale = nn.Parameter(self.clip.logit_scale.clone())

        # 保存投影层的引用（forward 中需要用到）
        self.text_projection = self.clip.text_projection
        self.visual_projection = self.clip.visual_projection

        # 只注册 class_embeds 为 buffer（不持久化保存），bos 和 eos 在 construct_prompt 中即时获取
        class_embeds_init = self._init_class_embeds(classnames).clone().detach()
        self.register_buffer("class_embeds", class_embeds_init, persistent=False)

        self._init_prompt()
        self._init_weights()
        self._setup_modules_state()

        # 删除完整的 clip 模型引用，避免保存时的内存共享问题
        del self.clip

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

    def _init_weights(self):
        if self.gate:
            nn.init.constant_(self.gate[-2].bias, 3.0)

        for m in self.metanet.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _setup_modules_state(self):
        # 显式冻结 text 模块的所有参数
        for param in self.token_embedding.parameters():
            param.requires_grad = False
        for param in self.position_embedding.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.final_layer_norm.parameters():
            param.requires_grad = False
        for param in self.text_projection.parameters():
            param.requires_grad = False

        # 解冻 vision 模块
        for param in self.vision_embedding.parameters():
            param.requires_grad = True
        for param in self.vision_pre_layernorm.parameters():
            param.requires_grad = True
        for param in self.vision_post_layernorm.parameters():
            param.requires_grad = True
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
        for param in self.visual_projection.parameters():
            param.requires_grad = True

    def text_encoder_forward(self, prompt):
        """
        prompt: (B,n_class,n_ctx+16+2,text_dim)
        clip text model从token embedding之后的处理过程
        """
        prompt_viewed = prompt.view(
            prompt.shape[0] * prompt.shape[1], prompt.shape[2], -1
        )  # (B*n_class,n_ctx+16+2,text_dim)

        inputs_embeds = prompt_viewed
        # 手动构造 embeddings，避免保存 self.embedding 导致的重复引用
        position_ids = torch.arange(
            inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device
        )
        position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.size(0), -1)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

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
        ]  # (B*n_class,text_dim) 取eos位置的输出作为句子表示

        return pooled_output

    def vision_encoder_forward(self, pixel_value_S, pixel_value_T):
        """
        分别接受源域和目标域图像
        """
        # 源域处理
        # 1. 获得浅层风格特征并归一化
        hidden_states_S = self.vision_embedding(pixel_value_S)
        hidden_states_S = self.vision_pre_layernorm(hidden_states_S)
        hidden_states_S_temp = hidden_states_S
        for i in self.vision_encoder.layers[:4]:
            hidden_states_S_temp = i(
                hidden_states_S_temp, attention_mask=None, causal_attention_mask=None
            )[0]

        style_states_S = hidden_states_S_temp
        style_mean_S = style_states_S.mean(dim=1, keepdim=True)
        style_std_S = style_states_S.std(dim=1, keepdim=True) + 1e-6
        style_normed_S = (style_states_S - style_mean_S) / style_std_S

        # 2. 完整通过视觉编码器
        for i in self.vision_encoder.layers[4:]:
            hidden_states_S = i(
                hidden_states_S, attention_mask=None, causal_attention_mask=None
            )[0]
        pooled_output_S = hidden_states_S[:, 0, :]  # (B,vision_dim)
        last_hidden_states_S = self.vision_post_layernorm(pooled_output_S)

        # 目标域处理
        # 1. 获得浅层风格特征
        hidden_states_T = self.vision_embedding(pixel_value_T)
        hidden_states_T = self.vision_pre_layernorm(hidden_states_T)
        hidden_states_T_temp = hidden_states_T
        for i in self.vision_encoder.layers[:4]:
            hidden_states_T_temp = i(
                hidden_states_T_temp, attention_mask=None, causal_attention_mask=None
            )[0]

        style_states_T = hidden_states_T_temp
        style_mean_T = style_states_T.mean(dim=1, keepdim=True)
        style_std_T = style_states_T.std(dim=1, keepdim=True) + 1e-6

        # 2. 完整通过视觉编码器
        for i in self.vision_encoder.layers[4:]:
            hidden_states_T = i(
                hidden_states_T, attention_mask=None, causal_attention_mask=None
            )[0]
        pooled_output_T = hidden_states_T[:, 0, :]  # (B,vision_dim)
        last_hidden_states_T = self.vision_post_layernorm(pooled_output_T)

        # 风格自适应
        hidden_states_adapted = style_normed_S * style_std_T + style_mean_T

        hidden_states_adapted_temp = hidden_states_adapted
        for i in self.vision_encoder.layers[4:]:
            hidden_states_adapted_temp = i(
                hidden_states_adapted_temp,
                attention_mask=None,
                causal_attention_mask=None,
            )[0]
        last_hidden_states_adapted = hidden_states_adapted_temp

        pooled_output_adapted = last_hidden_states_adapted[:, 0, :]  # (B,vision_dim)
        last_hidden_states_adapted = self.vision_post_layernorm(pooled_output_adapted)

        gate_weights = self.gate(last_hidden_states_adapted)  # (B,1)
        fused_vision_feats = (
            gate_weights * last_hidden_states_S
            + (1 - gate_weights) * last_hidden_states_adapted
        )

        return (
            fused_vision_feats,
            last_hidden_states_S,
            last_hidden_states_T,
            last_hidden_states_adapted,  # 返回 adapted_feats 用于计算 gate 统计量
        )  # (B,vision_dim)

    def infer(self, pixel_values):
        """
        单图推理接口，用于评估（只走源域路径，不需要目标域）
        """
        # 完整通过视觉编码器
        hidden_states = self.vision_embedding(pixel_values)
        hidden_states = self.vision_pre_layernorm(hidden_states)
        for blk in self.vision_encoder.layers:
            hidden_states = blk(
                hidden_states, attention_mask=None, causal_attention_mask=None
            )[0]
        pooled = hidden_states[:, 0, :]  # (B, vision_dim)
        vision_feats = self.vision_post_layernorm(pooled)

        # 构造 prompt 并计算分类 logits
        b = vision_feats.shape[0]
        prompt = self.construct_prompt(vision_feats)  # (B,n_class,n_ctx+16+2,text_dim)
        text_feats = self.text_encoder_forward(prompt)  # (B*n_class,text_dim)
        text_feats = self.text_projection(text_feats)  # (B*n_class,text_dim)
        text_feats = text_feats.view(b, -1, self.text_dim)  # (B,n_class,text_dim)

        img_feats = self.visual_projection(vision_feats)  # (B,text_dim)
        img_feats = img_feats.unsqueeze(1)  # (B,1,text_dim)

        logits = F.normalize(img_feats, dim=-1) @ F.normalize(
            text_feats, dim=-1
        ).permute(
            0, 2, 1
        )  # (B,1,n_class)
        logits = self.logit_scale.exp() * logits  # scaled logits
        logits = logits.squeeze(1)  # (B,n_class)

        return logits

    def construct_prompt(self, vision_feats):
        b = vision_feats.shape[0]
        n_class = self.class_embeds.shape[0]
        pi = self.metanet(vision_feats).unsqueeze(1)  # (B,1,text_dim)
        ctx = self.ctx.expand(b, -1, -1)  # (B,n_ctx,text_dim)
        ctx_pi = ctx + pi  # (B,n_ctx,text_dim) 广播
        ctx_pi = ctx_pi.unsqueeze(1).expand(
            b, n_class, -1, -1
        )  # (B,n_class,n_ctx,text_dim)
        prompt = torch.cat(
            [ctx_pi, self.class_embeds.unsqueeze(0).expand(b, -1, -1, -1)], dim=2
        )  # (B,n_class,n_ctx+16,text_dim)

        # 即时从 token_embedding 获取 bos 和 eos，避免内存共享问题
        bos_embed = self.token_embedding.weight[self.tokenizer.bos_token_id]
        eos_embed = self.token_embedding.weight[self.tokenizer.eos_token_id]

        bos = (
            bos_embed.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(b, n_class, -1, -1)
        )  # (B,n_class,1,text_dim)
        eos = (
            eos_embed.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(b, n_class, -1, -1)
        )  # (B,n_class,1,text_dim)
        prompt = torch.cat([bos, prompt, eos], dim=2)  # (B,n_class,n_ctx+16+2,text_dim)
        return prompt

    def forward(
        self,
        pixel_values_S=None,
        pixel_values_T=None,
        pixel_values=None,
        labels=None,
        domain_ids=None,
    ):
        # 评估模式：只传 pixel_values（用于推理）
        if pixel_values is not None:
            logits = self.infer(pixel_values)
            return GSPAOutput(logits=logits)

        # 训练模式：传 pixel_values_S 和 pixel_values_T
        fused_vision_feats, source_feats, target_feats, adapted_feats = (
            self.vision_encoder_forward(pixel_values_S, pixel_values_T)
        )

        # 计算分类 logits
        b = fused_vision_feats.shape[0]
        prompt_S = self.construct_prompt(
            fused_vision_feats
        )  # (B,n_class,n_ctx+16+2,text_dim)

        text_feats_S = self.text_encoder_forward(prompt_S)  # (B*n_class,text_dim)
        text_feats_S = self.text_projection(text_feats_S)  # (B*n_class,text_dim)
        text_feats_S = text_feats_S.view(b, -1, self.text_dim)  # (B,n_class,text_dim)

        image_feats_S = self.visual_projection(fused_vision_feats)  # (B,text_dim)
        image_feats_S = image_feats_S.unsqueeze(1)  # (B,1,text_dim)

        logits_task = F.normalize(image_feats_S, dim=-1) @ F.normalize(
            text_feats_S, dim=-1
        ).permute(
            0, 2, 1
        )  # (B,1,n_class)
        logits_task = self.logit_scale.exp() * logits_task  # scaled logits
        logits_task = logits_task.squeeze(1)  # (B,n_class)

        # 计算对齐logits
        # 统一反转 target_feats，确保 ViT 和 MetaNet 都受 GRL 影响
        target_feats_reversed = grad_reverse(
            target_feats, 1.0
        )  # 反转梯度：ViT 最大化熵（不确定），MetaNet 最小化熵（判别性）

        prompt_T = self.construct_prompt(
            target_feats_reversed
        )  # (B,n_class,n_ctx+16+2,text_dim)
        text_feats_T = self.text_encoder_forward(prompt_T)  # (B*n_class,text_dim)
        text_feats_T = self.text_projection(text_feats_T)  # (B*n_class,text_dim)
        text_feats_T = text_feats_T.view(b, -1, self.text_dim)  # (B,n_class,text_dim)

        image_feats_T = self.visual_projection(
            target_feats_reversed
        )  # (B,text_dim) 修复：使用反转后的特征
        image_feats_T = image_feats_T.unsqueeze(1)  # (B,1,text_dim)

        logits_align = F.normalize(image_feats_T, dim=-1) @ F.normalize(
            text_feats_T, dim=-1
        ).permute(
            0, 2, 1
        )  # (B,1,n_class)
        logits_align = self.logit_scale.exp() * logits_align  # scaled logits
        logits_align = logits_align.squeeze(1)  # (B,n_class)

        probs_a = F.softmax(logits_align, dim=-1)  # (B,n_class)
        loss_align = (
            -(probs_a * torch.log(probs_a + 1e-6)).sum(dim=-1).mean()
        )  # 目标域预测分布的熵作为对齐损失

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss_task = loss_fct(logits_task, labels)

            # 渐进式对齐损失权重（从训练参数中获取，如果没有则默认为 1.0）
            align_weight = getattr(self, "_align_weight", 1.0)
            loss_align_weighted = align_weight * loss_align  # 加权后的对齐损失
            loss_total = loss_task + loss_align_weighted

            with torch.no_grad():
                # 修改：使用 adapted_feats 计算 gate 统计量，与实际使用的输入一致
                gate_vals = self.gate(adapted_feats)
                gate_mean = gate_vals.mean().item()
                gate_std = gate_vals.std().item()

            return GSPAOutput(
                loss=loss_total,
                logits=None,
                loss_task=loss_task.detach(),
                loss_align=loss_align_weighted.detach(),  # 返回加权后的损失
                gate_mean=gate_mean,
                gate_std=gate_std,
            )

        else:
            return GSPAOutput(
                logits=logits_task,
            )


# 测试
if __name__ == "__main__":
    classnames = ["cat", "dog", "car", "bus"]
    model = ConditionalPromptLearner(model_name, classnames, n_ctx=4)
    processor = CLIPProcessor.from_pretrained(model_name)
    dummy_image_S = torch.randn(1, 3, 224, 224)
    dummy_image_T = torch.randn(1, 3, 224, 224)
    logits = model(dummy_image_S, dummy_image_T)  # (1,n_class)
    print(logits)
