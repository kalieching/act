import torch
import torch.nn as nn
import math

class CVAEEncoder(nn.Module):
    def __init__(
        self,
        action_dim,
        joint_dim,
        d_model,
        nhead,
        num_layers,
        chunk_size,
        latent_dim,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model

        self.action_proj = nn.Linear(action_dim, d_model)
        self.joint_proj = nn.Linear(joint_dim, d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.register_buffer(
            'pos_enc',
            sinusoidal_pos_encoding(chunk_size + 2, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)