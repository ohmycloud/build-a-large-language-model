import torch

# 将当前文本截断至支持的长度。如果大语言模型仅支持 5 个词元, 但此时文本长度为 10,
# 则只有最后 5 个词元会被用作输入文本
# idx 是当前文本的索引数组, 其形状为 (batch_size, n_tokens)
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个输出的内容, 因此其形状会从 (batch, n_token, vocab_size) 变为 (batch, vocab_size)
        logits = logits[:, -1, :]
        # probas 的形状为 (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)
        # idx_next 的形状为 (batch, 1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # 将计算出的下一个字符的索引添加到数组中, 此时 idx 的形状变为 (batch, n_token + 1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
