import torch
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