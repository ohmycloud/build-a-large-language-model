import torch
from compact_self_attention import SelfAttention_v1

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66],  # journey  (x^2)
    [0.57, 0.85, 0.64],  # starts   (x^3)
    [0.22, 0.58, 0.33],  # with     (x^4)
    [0.77, 0.25, 0.10],  # one      (x^5)
    [0.05, 0.80, 0.55]]  # step     (x^6)
)

torch.manual_seed(123)
d_in = inputs.shape[1]
d_out = 2
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
