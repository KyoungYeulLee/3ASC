"""Transformer building blocks of the MIL encoder."""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise feed-forward network of an encoder block.

    Args:
        hidden_dim (int): Size of the input and output dimension.
        ff_dim (int): Size of the intermediate dimension.
        dropout (float): Dropout probability.
    """

    def __init__(self, hidden_dim: int, ff_dim: int, dropout: float):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # With GELU, normal(std=0.02) initialisation trains more stably than
        # the Xavier default.
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (*, hidden_dim).

        Returns:
            torch.Tensor: Output tensor of shape (*, hidden_dim).
        """
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class EncoderBlock(nn.Module):
    """Pre-norm transformer encoder block.

    Args:
        hidden_dim (int): Size of the hidden dimension.
        num_heads (int): Number of attention heads.
        ff_dim (int): Size of the feed-forward intermediate dimension.
        dropout (float): Dropout probability.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = FeedForward(hidden_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        x_identity = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = x_identity + self.dropout(x)

        x_identity = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x_identity + self.dropout(x)

        return x


class Encoder(nn.Module):
    """Stack of encoder blocks.

    Args:
        hidden_dim (int): Size of the hidden dimension.
        num_heads (int): Number of attention heads.
        num_blocks (int): Number of encoder blocks.
        ff_dim (int): Size of the feed-forward intermediate dimension.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the input through every block.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        for block in self.encoder_blocks:
            x = block(x)

        return x


class AttentionFeatureProjection(nn.Module):
    """Embed the feature vector of a variant into one token.

    Each scalar feature is expanded over a bank of learned radial basis
    functions, projected, and the resulting per-feature tokens are mixed by
    self-attention. The CLS token of that mix is the variant embedding, so the
    embedder can model interactions between features rather than treating the
    feature vector as a flat linear input.

    Args:
        n_features (int): Number of input features.
        rbf_dim (int): Number of RBF centers per feature.
        output_dim (int): Size of the variant embedding.
    """

    def __init__(self, n_features: int, rbf_dim: int, output_dim: int):
        super().__init__()
        self.n_features = n_features
        self.rbf_dim = rbf_dim

        # RBF centers and scales for all features at once
        self.rbf_centers = nn.Parameter(
            torch.randn(n_features, rbf_dim)
        )  # [n_features, rbf_dim]
        self.rbf_scales = nn.Parameter(
            torch.ones(n_features, rbf_dim)
        )  # [n_features, rbf_dim]

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, output_dim)
        )  # [1, 1, output_dim]
        self.projection = nn.Linear(
            n_features * rbf_dim, n_features * output_dim
        )  # [n_features * rbf_dim -> n_features * output_dim]
        self.attention = nn.MultiheadAttention(
            output_dim, num_heads=4, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of variants.

        Args:
            x (torch.Tensor): Input tensor of shape
                [batch_size, n_variants, n_features].

        Returns:
            torch.Tensor: Output tensor of shape
                [batch_size, n_variants, output_dim].
        """
        batch_size, n_variants = x.shape[0], x.shape[1]

        # [batch_size, n_variants, n_features, 1]
        x_expanded = x.unsqueeze(-1)

        # [batch_size, n_variants, n_features, rbf_dim]
        rbf_out = torch.exp(
            -self.rbf_scales
            * (x_expanded - self.rbf_centers.unsqueeze(0).unsqueeze(0)) ** 2
        )

        # [batch_size, n_variants, n_features * rbf_dim]
        rbf_out = rbf_out.view(
            rbf_out.shape[0], rbf_out.shape[1], rbf_out.shape[2] * rbf_out.shape[3]
        )

        # [batch_size, n_variants, n_features * output_dim]
        projected = self.projection(rbf_out)

        # [batch_size, n_variants, n_features, output_dim]
        projected = projected.view(
            projected.shape[0],
            projected.shape[1],
            self.n_features,
            projected.shape[2] // self.n_features,
        )

        # [batch_size, n_variants, n_features + 1, output_dim]
        cls_tokens = self.cls_token.expand(batch_size, n_variants, 1, -1)
        projected = torch.cat([cls_tokens, projected], dim=2)

        # [batch_size * n_variants, n_features + 1, output_dim]
        projected = projected.view(-1, projected.shape[2], projected.shape[3])

        # [batch_size * n_variants, n_features + 1, output_dim]
        attended, _ = self.attention(projected, projected, projected)

        # [batch_size, n_variants, n_features + 1, output_dim]
        attended = attended.view(
            batch_size, n_variants, attended.shape[1], attended.shape[2]
        )

        # [batch_size, n_variants, output_dim]
        return attended[:, :, 0]
