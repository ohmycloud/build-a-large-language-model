import re
from typing import List
from simple_tokenizer_v1 import SimpleTokenizerV1

def tokenizer(text: str) -> List:
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item for item in result if item.strip()]
    return result

def vocabulary() -> dict:
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        preprocessed = tokenizer(raw_text)
        all_words = sorted(set(preprocessed))
        vocabulary_size = len(all_words)
        print(vocabulary_size)
        vocab = {token:integer for integer, token in enumerate(all_words)}
        return vocab

the_tokenizer = SimpleTokenizerV1(vocabulary())
text = """It's the last he painted, you know,"
       Mrs. Gisburn said with pardonable pride."""
ids = the_tokenizer.encode(text)
print(ids)
print(the_tokenizer.decode(ids))

text = "Hello, do you like tea?"
print(the_tokenizer.encode(text))
