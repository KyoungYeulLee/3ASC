"""The 3ASC 3.0 (MIL-GFE) network.

Two towers see the same bag of variants. The MIL tower embeds the numeric
feature vector of every variant and lets the variants attend to each other; the
GFE tower cross-attends the patient's symptom embedding against the disease
embedding of every variant. The GFE output is fused into the variant embeddings
of the MIL tower - only for variants whose disease is known - before the
transformer runs, so the ranking of a variant depends both on its own evidence
and on how well its disease matches the patient's phenotype.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from model.layers import AttentionFeatureProjection, Encoder


class SelfAttentionEmbedder(nn.Module):
    """Embed each variant type with its own feature-attention projection.

    Args:
        n_features (dict[str, int]): Number of features per variant type.
        rbf_dim (int): Number of RBF centers per feature.
        embedding_dim (int): Size of a variant embedding.
    """

    def __init__(
        self,
        n_features: dict[str, int],
        rbf_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        for variant_type, n_variant_features in n_features.items():
            setattr(
                self,
                f"{variant_type}_embedder",
                AttentionFeatureProjection(
                    n_features=n_variant_features,
                    rbf_dim=rbf_dim,
                    output_dim=embedding_dim,
                ),
            )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed every variant type and concatenate the results.

        Args:
            x (dict[str, torch.Tensor]): Feature tensor per variant type the
                patient has variants of, each of shape
                (batch_size, n_variants, n_features).

        Returns:
            torch.Tensor: Variant embeddings of shape
                (batch_size, n_all_variants, embedding_dim), ordered by variant
                type.
        """
        return torch.cat(
            [
                getattr(self, f"{variant_type}_embedder")(variant_x)
                for variant_type, variant_x in x.items()
            ],
            dim=1,
        )


class MilEncoder(nn.Module):
    """Transformer over the variants of one patient.

    Args:
        variant_embedder (nn.Module): Module embedding the variant features.
        embedding_dim (int): Size of a variant embedding.
        gfe_hidden_dim (int): Size of a GFE feature, fused into the embeddings.
        num_heads (int): Number of attention heads.
        num_blocks (int): Number of transformer blocks.
        ff_dim (int): Size of the feed-forward intermediate dimension.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        variant_embedder: nn.Module,
        embedding_dim: int,
        gfe_hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.variant_embedder = variant_embedder
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        self.gfe_hidden_dim = gfe_hidden_dim
        self.gfe_fusion_proj = nn.Sequential(
            nn.Linear(self.embedding_dim + self.gfe_hidden_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
            nn.ReLU(),
        )

        self.encoder = Encoder(
            hidden_dim=self.embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ff_dim=ff_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: dict[str, torch.Tensor],
        gfe_feature: torch.Tensor,
        gfe_known_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embed, fuse the GFE features and run the transformer.

        Args:
            x (dict[str, torch.Tensor]): Feature tensor per variant type.
            gfe_feature (torch.Tensor): GFE features of shape
                (batch_size, n_all_variants, gfe_hidden_dim).
            gfe_known_mask (torch.Tensor): Which variants have a known disease,
                of shape (batch_size, n_all_variants). Variants without one keep
                their unfused embedding.

        Returns:
            torch.Tensor: Encoded tokens of shape
                (batch_size, n_all_variants + 1, embedding_dim); position 0 is
                the CLS token summarising the bag.
        """
        embedded_variant_features = self.variant_embedder(x)

        fused = self.gfe_fusion_proj(
            torch.cat([embedded_variant_features, gfe_feature], dim=-1)
        )
        embedded_variant_features = torch.where(
            gfe_known_mask.unsqueeze(-1), fused, embedded_variant_features
        )

        x = torch.cat([self.cls_token, embedded_variant_features], dim=1)

        return self.encoder(x)


class GFERelationalInteractionModel(nn.Module):
    """Cross-attention between the patient phenotype and each variant disease.

    Args:
        patient_embedding_dim (int): Size of the patient symptom embedding.
        variant_embedding_dim (int): Size of a disease embedding.
        hidden_dim (int): Size of the GFE feature.
        num_attention_heads (int): Number of attention heads.
        num_layers (int): Number of cross-attention layers.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        patient_embedding_dim: int,
        variant_embedding_dim: int,
        hidden_dim: int,
        num_attention_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.patient_embedding_dim = patient_embedding_dim
        self.variant_embedding_dim = variant_embedding_dim
        self.hidden_dim = hidden_dim

        self.patient_projection = nn.Linear(patient_embedding_dim, hidden_dim)
        self.variant_projection = nn.Linear(variant_embedding_dim, hidden_dim)

        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norms1 = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        self.feedforward_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norms2 = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        patient_embeddings: torch.Tensor,
        variant_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Attend every variant disease against the patient phenotype.

        Args:
            patient_embeddings (torch.Tensor): Patient symptom embedding of
                shape (batch_size, 1, patient_embedding_dim).
            variant_embeddings (torch.Tensor): Disease embedding per variant, of
                shape (batch_size, n_variants, variant_embedding_dim).

        Returns:
            torch.Tensor: GFE features of shape
                (batch_size, n_variants, hidden_dim).
        """
        patient_features = self.patient_projection(
            patient_embeddings
        )  # [batch_size, 1, hidden_dim]
        variant_features = self.variant_projection(
            variant_embeddings
        )  # [batch_size, n_variants, hidden_dim]

        for attention, layer_norm1, feedforward, layer_norm2 in zip(
            self.attention_layers,
            self.layer_norms1,
            self.feedforward_layers,
            self.layer_norms2,
            strict=True,
        ):
            attn_output, _ = attention(
                variant_features, patient_features, patient_features
            )
            variant_features = layer_norm1(variant_features + self.dropout(attn_output))

            ff_output = feedforward(variant_features)
            variant_features = layer_norm2(variant_features + self.dropout(ff_output))

        return variant_features


class MILGFEModel(nn.Module):
    """MIL transformer stacked on top of the GFE cross-attention tower.

    Args:
        variant_embedder (nn.Module): Module embedding the variant features.
        embedding_dim (int): Size of a variant embedding.
        patient_embedding_dim (int): Size of the patient symptom embedding.
        variant_embedding_dim (int): Size of a disease embedding.
        gfe_hidden_dim (int): Size of a GFE feature.
        gfe_num_layers (int): Number of GFE cross-attention layers.
        num_heads (int): Number of attention heads, in both towers.
        num_blocks (int): Number of transformer blocks in the MIL tower.
        ff_dim (int): Size of the feed-forward intermediate dimension.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        variant_embedder: nn.Module,
        embedding_dim: int,
        patient_embedding_dim: int,
        variant_embedding_dim: int,
        gfe_hidden_dim: int,
        gfe_num_layers: int,
        num_heads: int,
        num_blocks: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mil_encoder = MilEncoder(
            variant_embedder=variant_embedder,
            embedding_dim=self.embedding_dim,
            gfe_hidden_dim=gfe_hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.gfe_encoder = GFERelationalInteractionModel(
            patient_embedding_dim=patient_embedding_dim,
            variant_embedding_dim=variant_embedding_dim,
            hidden_dim=gfe_hidden_dim,
            num_attention_heads=num_heads,
            num_layers=gfe_num_layers,
            dropout=dropout,
        )

        self.bag_classifier = nn.Linear(in_features=self.embedding_dim, out_features=1)

        self.unified_instance_classifier = nn.Sequential(
            nn.Linear(
                in_features=self.embedding_dim,
                out_features=(self.embedding_dim // 2),
            ),
            nn.ReLU(),
            nn.Linear(in_features=(self.embedding_dim // 2), out_features=1),
        )

    def forward(
        self,
        x: dict[str, torch.Tensor],
        v_embeddings: torch.Tensor,
        p_embeddings: torch.Tensor,
        gfe_known_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score the bag and each of its variants.

        Args:
            x (dict[str, torch.Tensor]): Feature tensor per variant type.
            v_embeddings (torch.Tensor): Disease embedding per variant.
            p_embeddings (torch.Tensor): Patient symptom embedding.
            gfe_known_mask (torch.Tensor): Which variants have a known disease.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Bag logit of shape
                (batch_size, 1) and instance logits of shape
                (batch_size, n_all_variants).
        """
        gfe_feature = self.gfe_encoder(
            patient_embeddings=p_embeddings,
            variant_embeddings=v_embeddings,
        )
        feature = self.mil_encoder(x, gfe_feature, gfe_known_mask)

        cls_token = feature[:, 0]  # [batch_size, embedding_dim]
        bag_logit = self.bag_classifier(cls_token)  # [batch_size, 1]
        instances = feature[:, 1:]  # [batch_size, n_variants, embedding_dim]
        instance_logits = self.unified_instance_classifier(instances).squeeze(dim=-1)

        return bag_logit, instance_logits


def build_model(cfg: DictConfig) -> MILGFEModel:
    """Instantiate the model described by a configuration.

    Args:
        cfg (DictConfig): Full configuration; `features` fixes the input width
            of every variant embedder and `model.parameters` the architecture.

    Returns:
        MILGFEModel: The randomly initialised model.
    """
    parameters = cfg.model.parameters
    variant_embedder = SelfAttentionEmbedder(
        n_features={
            variant_type: len(cfg.features[variant_type])
            for variant_type in cfg.model.variant_type
        },
        rbf_dim=parameters.self_attention.rbf_dim,
        embedding_dim=parameters.n_hiddens,
    )

    return MILGFEModel(
        variant_embedder=variant_embedder,
        embedding_dim=parameters.n_hiddens,
        patient_embedding_dim=parameters.mil_gfe.patient_embedding_dim,
        variant_embedding_dim=parameters.mil_gfe.variant_embedding_dim,
        gfe_hidden_dim=parameters.mil_gfe.gfe_hidden_dim,
        gfe_num_layers=parameters.mil_gfe.gfe_num_layers,
        num_heads=parameters.transformer.num_heads,
        num_blocks=parameters.transformer.num_blocks,
        ff_dim=parameters.transformer.ff_dim,
        dropout=parameters.transformer.dropout,
    )
