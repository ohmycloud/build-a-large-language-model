import re
from typing import List

def tokenizer(text: str) -> List:
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item for item in result if item.strip()]
    return result

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    preprocessed = tokenizer(raw_text)
    all_words = sorted(set(preprocessed))
    vocabulary_size = len(all_words)
    print(vocabulary_size)
    vocab = {token:integer for integer, token in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 50:
            break
