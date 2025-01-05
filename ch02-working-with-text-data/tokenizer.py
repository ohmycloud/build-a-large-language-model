import re
from typing import List

def tokenizer(text: str) -> List:
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item for item in result if item.strip()]
    return result

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    print(tokenizer(raw_text))

print("Total number of character:", len(raw_text))
print(raw_text[:99])
