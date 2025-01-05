import re
from typing import List
from simple_tokenizer_v1 import SimpleTokenizerV1
from simple_tokenizer_v2 import SimpleTokenizerV2

def tokenizer(text: str) -> List:
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item for item in result if item.strip()]
    return result

def vocabulary() -> dict:
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        preprocessed = tokenizer(raw_text)
        all_tokens = sorted(list(set(preprocessed)))
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])
        vocab = {token:integer for integer, token in enumerate(all_tokens)}
        return vocab

def simple_tokenizer_v1():
    tokenizer = SimpleTokenizerV1(vocabulary())
    text = """It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))

    text = "Hello, do you like tea?"
    print(tokenizer.encode(text))

def simple_tokenizer_v2():
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)
    tokenizer = SimpleTokenizerV2(vocabulary())
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))

if __name__ == '__main__':
    simple_tokenizer_v2()
