import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66],  # journey  (x^2)
    [0.57, 0.85, 0.64],  # starts   (x^3)
    [0.22, 0.58, 0.33],  # with     (x^4)
    [0.77, 0.25, 0.10],  # one      (x^5)
    [0.05, 0.80, 0.55]]  # step     (x^6)
)

# attention scores
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

# attention weights
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("注意力权重: ", attn_weights_2)
print("总和: ", torch.sum(attn_weights_2))

# context vector
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += x_i * attn_weights_2[i]
print("上下文向量: ", context_vec_2)

# compute all attention scores
attn_scores = inputs @ inputs.T
print("所有上下文向量: ", attn_scores)

# normalized context vectors
attn_weights = torch.softmax(attn_scores, dim=-1)
print("归一化上下文向量: ", attn_weights)

# verify the weighted sum
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum: ", row_2_sum)
print("All row sums: ", attn_weights.sum(dim=-1))

# compute all context vectors
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
