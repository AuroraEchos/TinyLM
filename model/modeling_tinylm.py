from transformers import PretrainedConfig

class TinyLMargs(PretrainedConfig):
    model_type = "TinyLM"

    def __init__(
        self,

        hidden_size: int = 512,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        dropout: float = 0.0,
        flash_attn: bool = True,

        intermediate_size: int = None,
        hidden_act: str = "silu",

        rms_norm_eps: float = 1e-05,

        vocab_size: int = 6400,

        max_position_embeddings: int = 32768,
        rope_theta: int = 1000000.0,
        inference_rope_scaling: bool = False,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None

        
        

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from transformers.activations import ACT2FN

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)

    RMSNorm(x) = x * (scale / sqrt(mean(x^2) + eps))

    Design choices:
    - no bias (matches LLaMA / Qwen / Mistral)
    - only a single scale parameter (parameter-efficient)
    - eps added inside sqrt for numerical stability (matches T5 / Qwen)
    - applied before attention / FFN (matches LLaMA / Mistral)
    - no centering (matches T5 / LLaMA / Mistral)
    - can be used with mixed precision without loss of stability
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms = norm_x / math.sqrt(self.dim)
        x_normed = x / (rms + self.eps)
        return x_normed * self.scale
    
class Attention(nn.Module):
    """
    Causal Self-Attention with:
      - Multi-Head Attention (MHA)
      - Grouped Query Attention (GQA)
      - Rotary Positional Embedding (RoPE)
      - KV Cache (past_key_value / present_key_value)
      - Optional Flash Attention (PyTorch SDPA)
    """

    def __init__(self, args: TinyLMargs):
        super().__init__()

        # Head argsuration
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads          # H
        self.n_local_kv_heads = self.num_key_value_heads       # H_kv
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        self.head_dim = args.hidden_size // args.num_attention_heads

        # Projections
        self.q_proj = nn.Linear(
            args.hidden_size,
            self.n_local_heads * self.head_dim,
            bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.n_local_kv_heads * self.head_dim,
            bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.n_local_kv_heads * self.head_dim,
            bias=False
        )
        self.o_proj = nn.Linear(
            self.n_local_heads * self.head_dim,
            args.hidden_size,
            bias=False
        )

        # Regularization
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # Flash Attention availability
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attn
        )

    def forward(
        self,
        x: torch.Tensor,  # [B, T, hidden_size]
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape

        # 1. Linear projections
        q = self.q_proj(x)  # [B, T, H*D]
        k = self.k_proj(x)  # [B, T, H_kv*D]
        v = self.v_proj(x)  # [B, T, H_kv*D]

        q = q.view(B, T, self.n_local_heads, self.head_dim)
        k = k.view(B, T, self.n_local_kv_heads, self.head_dim)
        v = v.view(B, T, self.n_local_kv_heads, self.head_dim)

        # 2. RoPE (applied before KV cache concat)
        cos, sin = position_embeddings  # [T, D]
        cos = cos.unsqueeze(1)          # [T, 1, D]
        sin = sin.unsqueeze(1)

        def rotate_half(x):
            return torch.cat(
                (-x[..., self.head_dim // 2:], x[..., : self.head_dim // 2]),
                dim=-1,
            )

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # 3. KV cache: concat past and current
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present_key_value = (k, v) if use_cache else None

        # 4. GQA expansion + transpose for attention
        q = q.transpose(1, 2)  # [B, H, T, D]

        k = (
            k[:, :, :, None, :]
            .expand(B, k.size(1), self.n_local_kv_heads, self.n_rep, self.head_dim)
            .reshape(B, k.size(1), self.n_local_heads, self.head_dim)
            .transpose(1, 2)
        )

        v = (
            v[:, :, :, None, :]
            .expand(B, v.size(1), self.n_local_kv_heads, self.n_rep, self.head_dim)
            .reshape(B, v.size(1), self.n_local_heads, self.head_dim)
            .transpose(1, 2)
        )

        # 5. Attention computation
        if self.flash and past_key_value is None:
            # Flash Attention: safe for training / prefill
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Standard attention (decode-safe)
            attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            T_kv = k.size(-2)
            causal_mask = torch.triu(
                torch.full((T, T_kv), float("-inf"), device=attn_scores.device),
                diagonal=1 + (T_kv - T),
            )
            attn_scores = attn_scores + causal_mask

            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask[:, None, None, :]

            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            attn_output = attn_probs @ v

        # 6. Output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, -1)

        out = self.o_proj(attn_output)
        out = self.resid_dropout(out)

        return out, present_key_value

class FeedForward(nn.Module):
    """
    SwiGLU-style FeedForward Network (LLaMA-style)

    FFN(x) = W_down( act(W_gate(x)) ⊙ W_up(x) )

    Design choices:
    - intermediate_size ≈ 8/3 * hidden_size (parameter-efficient)
    - no bias (matches LLaMA / Qwen / Mistral)
    - dropout applied after down projection
    """

    def __init__(self, args: TinyLMargs):
        super().__init__()

        if args.intermediate_size is None:
            inter = int(args.hidden_size * 8 / 3)
            # round up to multiple of 64 for GPU efficiency
            inter = 64 * ((inter + 63) // 64)
            args.intermediate_size = inter

        self.gate_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            args.intermediate_size, args.hidden_size, bias=False
        )

        self.act_fn = ACT2FN[args.hidden_act]
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        out = self.down_proj(gate * up)
        return self.dropout(out)

def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE frequencies for a given dimension and sequence length

    Returns:
    - cos: [end, dim] cosine frequencies
    - sin: [end, dim] sine frequencies

    Design choices:
    - base frequency of 10^6 (matches LLaMA / Mistral)
    - optional scaling for longer sequences (not used in TinyLM)
    """
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

class TinyLMBlock(nn.Module):
    """
    Single Transformer block for TinyLM

    Design choices:
    - Pre-LN (matches LLaMA / Mistral)
    - Attention -> FFN order (matches LLaMA / Mistral)
    - No bias in projections (matches LLaMA / Qwen / Mistral)
    """

    def __init__(self, layer_id: int, args: TinyLMargs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.attn = Attention(args)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        
        self.ffn = FeedForward(args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states, present_key_value = self.attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask
        )

        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.ffn(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value
    
class TinyLMModel(nn.Module):
    """
    TinyLMModel: Transformer Backbone (Causal)

    Responsibilities:
    -----------------
    - Token embedding
    - Transformer blocks (Attention + FFN)
    - RoPE position embedding slicing
    - KV cache propagation (per-layer)

    This class DOES NOT:
    --------------------
    - Compute logits
    - Compute loss
    - Know anything about labels
    """

    def __init__(self, args: TinyLMargs):
        super().__init__()
        self.args = args

        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

        self.layers = nn.ModuleList(
            [TinyLMBlock(layer_id=i, args=args) for i in range(args.num_hidden_layers)]
        )

        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=args.hidden_size // args.num_attention_heads,
            end=args.max_position_embeddings,
            rope_base=args.rope_theta,
            rope_scaling=args.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Returns
        -------
        hidden_states : [B, T, hidden_size]
        present_key_values : list[(k, v)] or None
        """

        B, T = input_ids.shape

        # Normalize KV cache format
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        # Determine RoPE start position
        start_pos = (
            past_key_values[0][0].shape[1]
            if past_key_values[0] is not None
            else 0
        )

        # Token embedding
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)

        # Slice RoPE embeddings
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + T],
            self.freqs_sin[start_pos : start_pos + T],
        )

        present_key_values = [] if use_cache else None

        # Transformer blocks
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present_kv = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            if use_cache:
                present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)

        return hidden_states, present_key_values

class TinyLMForCausalLM(nn.Module):
    """
    TinyLMForCausalLM: Causal Language Modeling Wrapper

    Responsibilities:
    -----------------
    - Add LM head on top of TinyLMModel
    - Tie embedding and LM head weights
    - Compute causal language modeling loss
    """

    def __init__(self, args: TinyLMargs):
        super().__init__()
        self.args = args

        self.model = TinyLMModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        # Weight tying (important for small models)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        """
        Returns
        -------
        If labels is None:
            logits, past_key_values
        If labels is not None:
            loss, logits, past_key_values
        """

        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Standard causal LM shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return loss, logits, past_key_values

@torch.no_grad()
def decode(
    model: TinyLMForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
):
    """
    input_ids: [B, T]  (prompt)
    returns:   [B, T + max_new_tokens]
    """
    model.eval()

    past_key_values = None
    generated = input_ids

    for _ in range(max_new_tokens):
        # Only feed the last token if KV cache exists
        if past_key_values is None:
            inp = generated
        else:
            inp = generated[:, -1:]

        _, logits, past_key_values = model(
            input_ids=inp,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # Take last token logits
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)

        # Greedy if temperature == 0
        if temperature == 0.0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

    return generated


if __name__ == "__main__":
    args = TinyLMargs()
    model = TinyLMForCausalLM(args)

    # Test forward pass
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = torch.tensor([[2, 3, 4, 5, 6]])
    loss, logits, _ = model(input_ids=input_ids, labels=labels)
    print(f"Loss: {loss.item()}")
    print(f"Logits shape: {logits.shape}")

    # Test decoding
    prompt = torch.tensor([[1, 2, 3]])
    generated = decode(model, prompt, max_new_tokens=5)
    print(f"Generated IDs: {generated}")


        




