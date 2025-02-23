# model.py

import torch
import torch.nn as nn
from transformers import LlamaModel
from config import Config


class MetaAttention(nn.Module):
    def __init__(self, attention_dim):
        super(MetaAttention, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, attention_dim)
        )

    def forward(self, A, C):
        # A: Original attention matrix (batch_size, heads, seq_length, seq_length)
        # C: Contextual adjustment term (batch_size, attention_dim)
        C = C.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, attention_dim)
        adjustment = self.transform(C)  # (batch_size, attention_dim)
        adjustment = adjustment.unsqueeze(-1)  # (batch_size, attention_dim, 1)
        A_meta = A + adjustment  # Broadcasting addition
        A_meta = torch.softmax(A_meta, dim=-1)
        return A_meta


class DynamicMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        super(DynamicMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = embed_dim

        self.q_linear = nn.Linear(embed_dim, num_heads * head_dim)
        self.k_linear = nn.Linear(embed_dim, num_heads * head_dim)
        self.v_linear = nn.Linear(embed_dim, num_heads * head_dim)
        self.out_linear = nn.Linear(num_heads * head_dim, embed_dim)

    def forward(self, Q, K, V, attention_mask=None):
        batch_size = Q.size(0)

        Q = self.q_linear(Q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)
        K = self.k_linear(K).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(V).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch, heads, seq, seq)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # (batch, heads, seq, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)  # (batch, seq, heads*head_dim)
        out = self.out_linear(context)  # (batch, seq, embed_dim)
        return out


class DualClassificationLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes=2):
        super(DualClassificationLayer, self).__init__()
        self.classifier1 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, H):
        y1 = self.classifier1(H)
        y2 = self.classifier2(H)
        return y1, y2


class LMAT_ND(nn.Module):
    def __init__(self, config):
        super(LMAT_ND, self).__init__()
        self.llama = LlamaModel.from_pretrained(config.TOKENIZER_PATH)
        self.meta_attention = MetaAttention(config.META_ATTENTION_DIM)
        self.dynamic_mha = DynamicMultiHeadAttention(
            embed_dim=self.llama.config.hidden_size,
            num_heads=config.DYNAMIC_MULTI_HEADS,
            head_dim=config.DYNAMIC_HEAD_DIM
        )
        self.dual_classifier = DualClassificationLayer(
            embed_dim=self.llama.config.hidden_size,
            hidden_dim=config.DUAL_CLASSIFIER_DIM
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq, hidden_size)

        # Self-Attention Mechanism
        # Assuming we extract attention matrices from Llama
        # This depends on the implementation of LlamaModel
        # Here, we assume we can get attention weights
        # For demonstration, we'll simulate attention matrix A
        A = torch.softmax(torch.randn(hidden_states.size(0), self.llama.config.num_attention_heads, hidden_states.size(1), hidden_states.size(1)), dim=-1).to(hidden_states.device)

        # Meta-Attention Mechanism
        # Compute context-aware transformation C
        # For simplicity, we'll use the mean of hidden states
        C = hidden_states.mean(dim=1)  # (batch, hidden_size)
        A_meta = self.meta_attention(A, C)  # (batch, heads, seq, seq)

        # Dynamic Multi-Head Attention
        H = self.dynamic_mha(hidden_states, hidden_states, hidden_states, attention_mask)
        H = self.dropout(H)

        # Dual-Classification Layer
        y1, y2 = self.dual_classifier(H.mean(dim=1))  # (batch, num_classes) each

        return y1, y2
