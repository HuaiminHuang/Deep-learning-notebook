import math
import json
import os
import time
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 参数配置文件
@dataclass
class DeepSeekV2Config:
    # 基础参数
    vocab_size: int = 100000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    
    # MLA 参数
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    kv_lora_rank: int = 512
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    rope_theta: float = 10000.0
    attention_bias: bool = False
    
    # MoE 参数
    expert_number: int = 8
    top_k: int = 2
    shared_expert_number: int = 2
    moe_load_balance_alpha: float = 0.01
    expert_dropout: float = 0.1
    
    # 训练参数
    batch_size: int = 4
    seq_len: int = 2048
    lr: float = 5e-5
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    total_steps: int = 100000
    grad_accum_steps: int = 1
    save_every: int = 1000
    
    # 其他参数
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    tie_word_embeddings: bool = True
    output_hidden_states: bool = False
    output_attentions: bool = False
    output_router_logits: bool = False

    # 日志和检查点
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    experiment_name: str = "llm_experiment"
    
    def save(self, path: str):
        """保存配置到JSON文件"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
    
    @classmethod
    def load(cls, path: str):
        """从JSON文件加载配置"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

# 基础组件
class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = max_position_embeddings

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_v2(q: torch.Tensor, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

# MoE组件
class FFNExpert(nn.Module):
    def __init__(self, hidden_dim, expert_dropout):
        super().__init__()
        mid_dim = hidden_dim * 8 // 3

        self.up = nn.Linear(hidden_dim, mid_dim, bias=False)
        self.down = nn.Linear(mid_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, mid_dim, bias=False)
        self.dropout = nn.Dropout(expert_dropout)

    def forward(self, x):
        output = self.dropout(
            self.down(
                F.silu(self.gate(x)) * self.up(x)
            )
        )
        return output

class MOERouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.expert_number)
        self.expert_number = config.expert_number
        self.top_k = config.top_k

    def forward(self, x):
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=1, dtype=torch.float)
        
        router_weights, selected_expert_indices = torch.topk(
            router_probs,
            self.top_k,
            dim=-1,
        )
        
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(x.dtype)
        
        expert_mask = F.one_hot(
            selected_expert_indices, 
            num_classes=self.expert_number,
        )
        
        expert_mask = expert_mask.permute(2, 1, 0)
        
        return router_logits, router_weights, selected_expert_indices, expert_mask

class SparseMOE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.top_k
        self.hidden_dim = config.hidden_size
        self.expert_number = config.expert_number

        self.experts = nn.ModuleList(
            [FFNExpert(config.hidden_size, config.expert_dropout) for _ in range(config.expert_number)]
        )
        self.router = MOERouter(config)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        hidden_states = x.view(-1, hidden_dim)
        
        router_logits, router_weights, _, expert_masks = self.router(hidden_states)
        
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            current_expert_mask = expert_masks[expert_idx]
            
            router_weight_idx, top_x = torch.where(current_expert_mask)
            
            current_states = hidden_states[top_x, :]
            current_states = expert_layer(current_states)
            
            current_token_router_weight = router_weights[top_x, router_weight_idx].unsqueeze(-1)
            current_hidden_states = current_states * current_token_router_weight
            
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        router_logits = router_logits.view(batch_size, seq_len, -1)
        
        return final_hidden_states, router_logits

class ShareExpertMOE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.moe_model = SparseMOE(config)
        self.shared_experts = nn.ModuleList([
            FFNExpert(config.hidden_size, config.expert_dropout) 
            for _ in range(config.shared_expert_number)
        ])

    def forward(self, x):
        sparse_moe_out, router_logits = self.moe_model(x)
        
        shared_experts_out = torch.stack(
            [expert(x) for expert in self.shared_experts], dim=0
        ).sum(dim=0)
        
        return sparse_moe_out + shared_experts_out, router_logits

# MLA注意力机制
class MLAV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.q_down_proj = nn.Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias=config.attention_bias,
        )
        self.q_down_layernorm = DeepseekV2RMSNorm(self.q_lora_rank)

        self.q_up_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.q_head_dim,
            bias=False,
        )

        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_down_layernorm = DeepseekV2RMSNorm(self.kv_lora_rank)
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        self.rotary_emb = DeepseekV2RotaryEmbedding(
            self.qk_rope_head_dim,
            self.max_position_embeddings,
            self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        compressed_kv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        # Query projection and split
        q = self.q_up_proj(self.q_down_layernorm(self.q_down_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Key/Value projection and split
        kv_seq_len = compressed_kv.size(1)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        
        k_pe = k_pe.view(bsz, kv_seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # Split kv_up_proj into heads
        kv_up_proj = self.kv_up_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
        q_absorb = kv_up_proj[:, :self.qk_nope_head_dim, :]
        out_absorb = kv_up_proj[:, self.qk_nope_head_dim:, :]

        # Apply RoPE
        cos, sin = self.rotary_emb(q_pe, seq_len=q_len)
        q_pe = apply_rotary_pos_emb_v2(q_pe, cos, sin, position_ids)

        # Project q_nope
        q_nope = torch.matmul(q_nope, q_absorb)

        # Attention score calculation
        attn_weights = (
            torch.matmul(q_pe, k_pe.mT)
            + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT)
        ) / math.sqrt(self.q_head_dim)

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Causal mask for decoder-only models
        causal_mask = torch.tril(
            torch.ones((q_len, kv_seq_len), device=attn_weights.device, dtype=torch.bool)
        )
        causal_mask = causal_mask.view(1, 1, q_len, kv_seq_len)
        attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))
        
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q_nope.dtype)

        # Compute attention output
        attn_output = torch.einsum("bhql,blc->bhqc", attn_weights, compressed_kv)
        attn_output = torch.matmul(attn_output, out_absorb.mT)

        # Merge heads and project
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Pre-LN architecture
        self.ln1 = DeepseekV2RMSNorm(config.hidden_size)
        self.attn = MLAV2(config)
        self.ln2 = DeepseekV2RMSNorm(config.hidden_size)
        self.moe = ShareExpertMOE(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        compressed_kv: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output, attn_weights = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            compressed_kv=compressed_kv,
        )
        hidden_states = residual + self.dropout(attn_output)

        # MoE FFN
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ffn_output, router_logits = self.moe(hidden_states)
        hidden_states = residual + self.dropout(ffn_output)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if output_router_logits:
            outputs += (router_logits,)
            
        return outputs

# 完整模型
class DeepSeekV2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        self.norm = DeepseekV2RMSNorm(config.hidden_size)
        
        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits

        # Prepare inputs
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        
        # Embed positions and tokens
        inputs_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        # Prepare compressed KV for MLA (for now, just use hidden states)
        compressed_kv = hidden_states
        
        # Initialize output containers
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        
        # Forward through layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                compressed_kv=compressed_kv,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                
            if output_router_logits and len(layer_outputs) > 2:
                all_router_logits = all_router_logits + (layer_outputs[2],)
        
        hidden_states = self.norm(hidden_states)
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Calculate MoE load balancing loss
        moe_loss = 0.0
        if output_router_logits and all_router_logits is not None:
            for router_logits in all_router_logits:
                moe_loss += self._calculate_moe_loss(router_logits)
        
        return {
            "logits": logits,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
            "router_logits": all_router_logits,
            "moe_loss": moe_loss,
        }
    
    def _calculate_moe_loss(self, router_logits):
        """Calculate MoE load balancing loss"""
        if router_logits is None:
            return 0.0
            
        router_logits = router_logits.view(-1, self.config.expert_number)
        probs = F.softmax(router_logits, dim=-1)
        
        # Calculate load balancing loss
        expert_mask = torch.gt(probs, 0).float()
        expert_fraction = expert_mask.mean(dim=0)
        router_fraction = probs.mean(dim=0)
        
        moe_loss = self.config.moe_load_balance_alpha * torch.sum(
            expert_fraction * router_fraction
        ) * self.config.expert_number
        
        return moe_loss

# 学习率调度器
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        
        # Set initial learning rates
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        
    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def _get_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.current_step / self.warmup_steps * self.optimizer.param_groups[0]['initial_lr']
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress)) * self.optimizer.param_groups[0]['initial_lr']
    
    def state_dict(self):
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
        }
    
    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']

# 训练器
class LLMTrainer:
    def __init__(self, model, config, train_dataset, val_dataset=None):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器和学习率调度器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
        
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, 
            config.warmup_steps, 
            config.total_steps
        )
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()
        
        # TensorBoard 记录器
        log_dir = os.path.join(config.log_dir, config.experiment_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # 训练状态
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        input_ids = batch.to(self.device)
        
        with torch.cuda.amp.autocast():
            outputs = self.model(input_ids)
            logits = outputs["logits"]
            
            # 计算语言建模损失
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            # 添加MoE负载均衡损失
            moe_loss = outputs.get("moe_loss", 0.0)
            total_loss = lm_loss + self.config.moe_load_balance_alpha * moe_loss
            
            # 梯度累积
            total_loss = total_loss / self.config.grad_accum_steps
        
        # 反向传播
        self.scaler.scale(total_loss).backward()
        
        # 记录损失到 TensorBoard
        self.writer.add_scalar('train/lm_loss', lm_loss.item(), self.global_step)
        self.writer.add_scalar('train/moe_loss', moe_loss.item(), self.global_step)
        self.writer.add_scalar('train/total_loss', total_loss.item() * self.config.grad_accum_steps, self.global_step)
        self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
        
        # 梯度累积步骤
        if (self.global_step + 1) % self.config.grad_accum_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return {
            "lm_loss": lm_loss.item(),
            "total_loss": total_loss.item() * self.config.grad_accum_steps,
            "moe_loss": moe_loss.item()
        }
    
    def validate(self):
        if self.val_dataset is None:
            return None
            
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_dataset:
                input_ids = batch.to(self.device)
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids)
                    logits = outputs["logits"]
                    
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)), 
                        shift_labels.view(-1)
                    )
                
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)
        
        avg_loss = total_loss / total_samples
        self.writer.add_scalar('val/loss', avg_loss, self.global_step)
        
        # 保存最佳模型
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(f"best_model.pt")
        
        return avg_loss
    
    def train(self):
        # 数据加载器
        def collate_fn(batch):
            max_len = min(max([b.size(0) for b in batch]), self.config.seq_len)
            input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
            for i, b in enumerate(batch):
                l = min(b.size(0), max_len)
                input_ids[i, :l] = b[:l]
            return input_ids
        
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        # 训练循环
        progress_bar = tqdm(total=self.config.total_steps, desc="Training")
        
        for epoch in range(100):  # 足够大的epoch数，通过total_steps控制
            for batch in train_loader:
                if self.global_step >= self.config.total_steps:
                    break
                
                # 训练步骤
                metrics = self.train_step(batch)
                
                # 更新进度条
                progress_bar.set_postfix({
                    "loss": f"{metrics['total_loss']:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                progress_bar.update(1)
                
                # 验证和保存检查点
                if self.global_step % 100 == 0:
                    val_loss = self.validate()
                    if val_loss is not None:
                        tqdm.write(f"Step {self.global_step}: val_loss = {val_loss:.4f}")
                
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
                
                self.global_step += 1
            
            if self.global_step >= self.config.total_steps:
                break
        
        progress_bar.close()
        self.writer.close()
        
        # 保存最终模型
        self.save_checkpoint("final_model.pt")
    
    def save_checkpoint(self, filename):
        checkpoint_path = os.path.join(self.config.checkpoint_dir, self.config.experiment_name, filename)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config)
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

# 使用示例
if __name__ == "__main__":
    # 创建配置
    config = DeepSeekV2Config()
    
    # 创建模型
    model = DeepSeekV2Model(config)
    
    # 创建示例数据集（需要根据实际情况实现）
    class ExampleDataset(Dataset):
        def __init__(self, size=1000, seq_len=2048, vocab_size=100000):
            self.data = [torch.randint(0, vocab_size, (seq_len,)) for _ in range(size)]
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = ExampleDataset(size=1000)
    val_dataset = ExampleDataset(size=100)
    
    # 创建训练器并开始训练
    trainer = LLMTrainer(model, config, train_dataset, val_dataset)
    trainer.train()