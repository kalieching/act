import torch
import torch.nn as nn
import math

def sinusoidal_pos_encoding(seq_len, d_model):
    """
    Transformer's self-attention is order-blind and sequence can be shuffled.
    Positional encoding fixes this by adding unique signal to each position.
    Model can tell token 0 from token 99.

    Sinusoidal encoding is a deterministic function of position, no learnable parameters.
    Each position is assigned a unique combination of sinusoids of different frquencies across d_model dimensions.
    Each dimension oscillates at different frequency.
    Unique fingerprint per position.
    Model can learn to attend to specific frequencies to extract relative or absolute position information.
    """
    # pe holds one row per position, one column per dimension of the model's token embeddings
    pe = torch.zeros(seq_len, d_model) 
    # must unsqueeze to make it column for broadcasting with div_term, float for sin/cos
    position = torch.arange(seq_len).unsqueeze(1).float()
    # controls frequency of each sin/cos pair
    # early dimensions -> high frequency (changes fast), later dimensions -> low frequency (changes slow)
    # together uniquely identify every position
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
    )
    
    pe[:, 0::2] = torch.sin(position * div_term) # even dims -> sin
    pe[:, 1::2] = torch.cos(position * div_term) # odd dims -> cos
    return pe.unsqueeze(0) # (1, seq_len, d_model), unsqueeze to make batch dimension for easy addition to token embeddings


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
    
    def forward(self, joint_state, actions):
        # B is batch size: how many training examples processed at once
        B = joint_state.shape[0]

        act_tokens = self.action_proj(actions)
        # unsqueeze to add sequence dimension to concatenate to other tokens
        joint_token = self.joint_proj(joint_state).unsqueeze(1)

        # create CLS token, one per example in batch, to hold summary of whole sequence after self-attention
        # expand copies it across the batch dimension
        # each batch starts with same CLS toekn, but after self-attention, each CLS 
        #    token will have different values because it attends to different joint states 
        #    and action sequences in each example
        cls = self.cls_token.expand(B, -1, -1)

        # feed this to transformer
        # [CLS | joint_state | action_0 | ... | action_99]
        # current robot state and action sequence about to execute
        seq = torch.cat([cls, joint_token, act_tokens], dim=1)

        seq = seq + self.pos_enc

        enc = self.transformer(seq)

        # after self-attention runs in the transformer, every token has attended to every other token
        # contains: given current robot joints, and full sequence of actions that follow, what is the overall style/character of this behavior?
        cls_out = enc[:, 0, :]

        # summary projected to mu and log_var to answer what kind of motion is happening
        # two separate linear layers read the CLS summary and outputs parameters of latent distribution
        # used in reparameterization trick to sample z
        mu = self.mu_head(cls_out)
        log_var = self.logvar_head(cls_out)

        return mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # eps ~ N(0, 1), same shape as std
        # this is z, this is passed to decoder as latent style variable that captures overall character of motion
        # sampled from distribution defined by mu and log_var
        return mu + eps * std

if __name__ == '__main__':
    encoder = CVAEEncoder(
        action_dim=14,
        joint_dim=14,
        d_model=256,
        nhead=8,
        num_layers=4,
        chunk_size=100,
        latent_dim=32,
    )

    B = 4
    joint_state = torch.randn(B, 14)
    actions     = torch.randn(B, 100, 14)

    mu, log_var = encoder(joint_state, actions)
    z = CVAEEncoder.reparameterize(mu, log_var)

    print(f"mu:      {mu.shape}")       # (4, 32)
    print(f"log_var: {log_var.shape}")  # (4, 32)
    print(f"z:       {z.shape}")        # (4, 32)
