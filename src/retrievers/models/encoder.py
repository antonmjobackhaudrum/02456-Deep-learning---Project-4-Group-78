import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -math.inf)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """A simple Multi-head attention layer."""
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None # store the attention maps
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class EncoderLayer(nn.Module):
    # pre-LN: x + attention(norm(x)), then x + feed_forward(norm(x))
    def __init__(self, emb_dim, n_heads, feed_forward_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(n_heads,emb_dim,dropout)
        self.norm1 = nn.LayerNorm(emb_dim)     
        self.norm2 = nn.LayerNorm(emb_dim)
        self.feed_forward = nn.Sequential(nn.Linear(emb_dim,feed_forward_dim), nn.ReLU(), nn.Linear(feed_forward_dim,emb_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_norm = self.norm1(x)
        x = x + self.dropout(self.attention(x_norm,x_norm,x_norm,mask))
        x_norm2 = self.norm2(x)
        x = x + self.dropout(self.feed_forward(x_norm2))
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    # From the course notebook (week 6, 6.1)
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    # Added vocab_size, pooling_type, max_len and dropout
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int,
                 vocab_size: int = 30522, pooling_type: str = 'cls', max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        assert emb_dim == n_heads * hidden_dim, 'hidden_dim is the per-head size: emb_dim must equal n_heads * hidden_dim'
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim =    emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.vocab_size = vocab_size
        self.pooling_type = pooling_type
        self.max_len = max_len
        self.dropout = dropout
        self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """
        self.embedding = nn.Embedding(self.vocab_size,self.emb_dim)
        self.positional_encoding = PositionalEncoding(self.emb_dim,self.dropout,self.max_len)
        self.layers = nn.ModuleList([EncoderLayer(self.emb_dim, self.n_heads, self.feed_forward_dim,self.dropout) for _ in range(self.n_layers)])   
        self.final_norm = nn.LayerNorm(self.emb_dim)

    def pool(self, x, mask):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        if self.pooling_type == 'cls':
            return x[:, 0, :]   # [CLS] is always first
        
        elif self.pooling_type == 'max':
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(mask==0,-math.inf)
            return x.max(dim=1).values
        
        elif self.pooling_type == 'mean':
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(mask==0,0)
            return x.sum(dim=1)/mask.sum(dim=1)

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """
        Run a forward pass on the model
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x,mask)
        x = self.final_norm(x)

        return self.pool(x,mask)
