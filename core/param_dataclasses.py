from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Dict, List, Optional

import torch


class PositionalEmbeddingType(Enum):
    REL = "core.models.positional_embeddings.RelativePositionBias"
    SINE = "core.models.positional_embeddings.ScaledSinusoidalEmbedding"
    ALIBI = "core.models.positional_embeddings.AlibiPositionalBias"
    ABS = "core.models.positional_embeddings.AbsolutePositionalEmbedding"
    SHAW = "core.models.positional_embeddings.ShawRelativePositionalEmbedding"


class MotionRep(Enum):
    FULL = "full"
    BODY = "body"
    HAND = "hand"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"


class TextRep(Enum):
    POOLED_TEXT_EMBED = "pooled_text_embed"
    FULL_TEXT_EMBED = "full_text_embed"


class AudioRep(Enum):
    ENCODEC = "encodec"
    LIBROSA = "librosa"
    WAV = "wav"


@dataclass
class AttentionParams:
    dim: int = 768
    heads: int = 8
    causal: bool = False
    qk_norm: bool = False
    qk_norm_scale: int = 8
    dropout: float = 0.0
    cross_attn_tokens_dropout: float = 0.0
    add_null_kv: bool = False
    flash: bool = False


@dataclass
class PositionalEmbeddingParams:
    dim: int = 768  # shaw , abs sine
    max_seq_len: int = 512  # shaw , abs
    causal: bool = False  # rel
    num_buckets: int = 32  # rel
    scale: float = 10.0  # rel
    heads: int = 8  # rel alibi
    total_heads: int = 8  # alibi
    theta = 10000  # sine
    max_distance = 128  # rel


@dataclass
class VQVAEOutput:
    decoded_motion: torch.Tensor
    quantized_motion: torch.Tensor = None
    indices: torch.Tensor = None
    commit_loss: torch.Tensor = None
    perplexity: torch.Tensor = None


@dataclass
class TranslationTransformerOutput:
    pred_motion: torch.Tensor
    recon_loss: torch.Tensor
    contact_loss: Optional[torch.Tensor] = None


@dataclass
class MotionBERTOutput:
    loss: torch.Tensor
    last_hidden_state: torch.Tensor
    logits: torch.Tensor


@dataclass
class ConformerParams:
    attention_params: AttentionParams
    positional_embedding_params: PositionalEmbeddingParams = None
    positional_embedding: PositionalEmbeddingType = None
    emb_dropout = 0.0
    conv_expansion_factor: int = 2
    conv_kernel_size: int = 31
    conv_causal: bool = False
    ff_mult: int = 4
    conv_causal: bool = False
    conv_dropout: float = 0.0
    ff_dropout: float = 0.0
    depth: int = 3


@dataclass
class TransformerParams:
    self_attention_params: AttentionParams
    cross_attention_params: AttentionParams = None
    positional_embedding_params: PositionalEmbeddingParams = None
    positional_embedding: PositionalEmbeddingType = PositionalEmbeddingType.SINE

    num_tokens: int = 1024
    dim_out: int = None
    depth: int = 12
    ff_mult: int = 4
    self_cond: bool = False
    add_mask_id: bool = True
    emb_dropout: float = 0.0
    cond_dropout: float = 0.0
    post_emb_norm: bool = False
    context_dim: int = 768
    style_dim: int = 768


@dataclass
class TranslationTransformerParams(TransformerParams):
    dim_out: int = 4
    audio_input_dim: int = 128
    text_input_dim: int = 768
    loss_fnc: str = "l1_smooth"
    fuse_method: Dict[str, List[str]] = field(
        default_factory=lambda: {"cross": ["audio"], "prepend": ["text"]}
    )


@dataclass
class MotionTokenizerParams:
    vocab_size: int = 1027
    pad_token_id: int = 1024
    cls_token_id: int = 1025
    mask_token_id: int = 1026
