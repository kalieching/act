import torch
import torch.nn as nn
from utils import sinusoidal_pos_encoding

class ACTDecoder(nn.Module):
    """
    Does the inverse of the encoder - takes observation and expands it back out into full action sequence.
    
    encoder memory (the "context"): z, joint_state, image_tokens
    <cross-attention>
    action queries (one per timestep)
    predicted_actions (one per timestep) 
    """
    def __init__(self, d_model, nhead, num_layers, chunk_size, action_dim, joint_dim, latent_dim):
        super().__init__()

        self.chunk_size = chunk_size
        self.d_model = d_model

        # project z and joint state to d_model so they can join image_tokens in the context sequence
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.joint_proj = nn.Linear(joint_dim, d_model) 

        self.register_buffer('query_pos',
                             sinusoidal_pos_encoding(chunk_size, d_model))
        
        # the transformer decoder: self-attention and cross-attention per layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # final projection: d_model -> action_dim, applied at every timestep
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, z, joint_state, image_tokens):
        B = z.shape[0]

        # build context: pack z, joint_state, image_tokens into one sequence
        z_tokens = self.z_proj(z).unsqueeze(1)
        js_tokens = self.joint_proj(joint_state).unsqueeze(1)
        memory = torch.cat([z_tokens, js_tokens, image_tokens], dim=1)

        # black queries and positional identity
        queries = torch.zeros(B, self.chunk_size, self.d_model, device=z.device)
        queries = queries + self.query_pos

        # self-attention among queries, then cross-attention to memory
        # repeated num_layers times with residual connections in between, each time attending to the original memory (not updated by attention)
        out = self.transformer(queries, memory)

        # project each timestep's d_model vector -> action_dim to get predicted action sequence
        return self.action_head(out)

if __name__ == '__main__':
    decoder = ACTDecoder(d_model=256, nhead=8, num_layers=7,
                         chunk_size=100, action_dim=14,
                         joint_dim=14, latent_dim=32)
    B = 4
    actions = decoder(torch.randn(B, 32),   # z
                      torch.randn(B, 14),   # joint_state
                      torch.randn(B, 98, 256))  # image_tokens
    print(actions.shape)  # (4, 100, 14)
