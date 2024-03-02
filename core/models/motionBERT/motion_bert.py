import math
from collections import namedtuple
from enum import Enum
from functools import partial, wraps

import torch
import torch.nn.functional as F
from core import (AttentionParams, ConformerParams, MotionBERTOutput,
                  MotionTokenizerParams, PositionalEmbeddingParams,
                  PositionalEmbeddingType)
from core.models.conformer import ConformerBlock
from core.models.utils import get_obj_from_str
from einops import rearrange, repeat
from packaging import version
from torch import einsum, nn
from torch.nn import CrossEntropyLoss, MSELoss


class BertOnlyMLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
        )

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, sequence_output):
        sequence_output = self.transform(sequence_output)
        prediction_scores = self.decoder(sequence_output)
        return prediction_scores


class BERTFORMER(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = MotionTokenizerParams()
        conformer_params = ConformerParams(
            attention_params=AttentionParams(
                dim=args.dim, heads=args.heads, dropout=args.dropout
            ),
            positional_embedding=PositionalEmbeddingType.REL,
            positional_embedding_params=PositionalEmbeddingParams(dim=args.dim),
            conv_kernel_size=args.conv_kernel_size,
            conv_expansion_factor=2,
            conv_dropout=args.dropout,
            ff_dropout=args.dropout,
            depth=args.depth,
        )

        self.is_abs_pos_emb = conformer_params.positional_embedding.name in [
            "ABS",
            "SINE",
        ]

        self.pos_emb = get_obj_from_str(conformer_params.positional_embedding.value)(
            conformer_params.positional_embedding_params
        )

        self.mlm_probability = args.mlm_probability

        self.motion_embeddings = nn.Embedding(
            self.tokenizer.vocab_size, args.dim, padding_idx=self.tokenizer.pad_token_id
        )

        self.post_emb_norm = nn.LayerNorm(args.dim)

        self.bert = ConformerBlock(conformer_params)
        self.cls = BertOnlyMLMHead(args.dim, self.tokenizer.vocab_size)

    def load(self, path):
        pkg = torch.load(path, map_location="cuda")
        self.load_state_dict(pkg["model"])

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids,
        input_embeds=None,
        attention_mask=None,
        labels=None,
        soft_labels=None,
        alpha=0,
        return_logits=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        if input_embeds is None:
            input_embeds = self.motion_embeddings(input_ids)

        if self.is_abs_pos_emb:
            input_embeds = input_embeds + self.pos_emb(input_embeds)

        input_embeds = self.post_emb_norm(input_embeds)

        outputs = self.bert(
            input_embeds,
            mask=attention_mask,
        )

        sequence_output = outputs
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.tokenizer.vocab_size), labels.view(-1)
            )

        if soft_labels is not None:
            loss_distill = -torch.sum(
                F.log_softmax(prediction_scores, dim=-1) * soft_labels, dim=-1
            )
            loss_distill = loss_distill[labels != -100].mean()
            masked_lm_loss = (1 - alpha) * masked_lm_loss + alpha * loss_distill

        return MotionBERTOutput(
            loss=masked_lm_loss,
            last_hidden_state=sequence_output,
            logits=prediction_scores,
        )
