from dataclasses import dataclass

@dataclass
class ACTConfig:
    # architecture
    d_model:    int   = 256
    nhead:      int   = 8
    enc_layers: int   = 4
    dec_layers: int   = 7
    latent_dim: int   = 32
    chunk_size: int   = 100

    # task — change these to match your robot
    action_dim: int   = 14   # 7 joints × 2 arms
    joint_dim:  int   = 14
    num_cams:   int   = 2

    # training
    batch_size:   int   = 8
    lr:           float = 1e-4
    weight_decay: float = 1e-4
    num_epochs:   int   = 100
    kl_weight:    float = 10.0