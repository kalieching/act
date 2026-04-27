import torch
import torch.nn as nn
from model.backbone import ImageBackbone
from model.cvae_encoder import CVAEEncoder
from model.transformer_decoder import ACTDecoder

class ACTPolicy(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.backbone = ImageBackbone(
            d_model=cfg.d_model,
            pretrained=True,
        )

        self.encoder = CVAEEncoder(
            action_dim=cfg.action_dim,
            joint_dim=cfg.joint_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.enc_layers,
            chunk_size=cfg.chunk_size,
            latent_dim=cfg.latent_dim,
        )

        self.decoder = ACTDecoder(
            d_model=cfg.d_model,    
            nhead=cfg.nhead,
            num_layers=cfg.dec_layers,
            chunk_size=cfg.chunk_size,
            action_dim=cfg.action_dim,
            joint_dim=cfg.joint_dim,
            latent_dim=cfg.latent_dim,
        )

        self.latent_dim = cfg.latent_dim

    def forward(self, joint_state, images, actions=None):
        B = joint_state.shape[0]
        image_tokens = self.backbone(images)  # (B, T, d_model)
        if self.training and actions is not None:
            mu, log_var = self.encoder(joint_state, actions) # runs forward function
            z = CVAEEncoder.reparameterize(mu, log_var)  # (B, latent_dim)
        else:
            mu = log_var = None
            z = torch.zeros(B, self.latent_dim, device=joint_state.device)

        pred_actions = self.decoder(z, joint_state, image_tokens) # runs forward function
        return pred_actions, mu, log_var

if __name__ == '__main__':
    from config import ACTConfig
    cfg = ACTConfig()
    model = ACTPolicy(cfg)

    B = 2
    joint_state = torch.randn(B, 14)
    images = torch.randn(B, 2, 3, 224, 224)
    actions = torch.randn(B, 100, 14)

    # training mode
    model.train()
    pred, mu, log_var = model(joint_state, images, actions)
    print(f"train pred: {pred.shape}") # (2, 100, 14)
    print(f"mu: {mu.shape}") # (2, 32)

    # inference mode
    model.eval()
    with torch.no_grad():
        pred, mu, log_var = model(joint_state, images)
    print(f"eval pred:  {pred.shape}") # (2, 100, 14)
    print(f"mu is None: {mu is None}") # True
    print("ACTPolicy OK")