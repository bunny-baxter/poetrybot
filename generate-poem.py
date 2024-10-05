# Generates a poem.

import argparse
import random
import re
import torch
import transformers

parser = argparse.ArgumentParser(prog = "Generate Poem")
parser.add_argument("--temp", type = float, default = 1.0, help = "model temperature")
args = parser.parse_args()

TOKENS_PER_LINE = 16
LINES_PER_STANZA = 4
STANZAS = 4

def _infer_once(model, text_input, temperature):
    with torch.no_grad():
        tokenized = tokenizer(text_input, return_tensors = "pt", padding = True)
    output = model(tokenized.input_ids, attention_mask = tokenized.attention_mask)
    next_token_probabilities = output.logits[0][-1]
    if temperature == 0.0:
        next_token = next_token_probabilities.argmax(0)
    else:
        if temperature != 1.0:
            next_token_probabilities = next_token_probabilities / temperature
        next_token = torch.multinomial(next_token_probabilities.softmax(0), 1)[0]
    return tokenizer.decode(next_token)

def generate(model, prompt, n_tokens, temperature):
    output = ""
    for _ in range(n_tokens):
        output += _infer_once(model, prompt + output, temperature)
    return output

EXTRA_WHITE_SPACE_REGEX = re.compile(r"\W\W+")

def normalize_whitespace(s):
    return re.sub(EXTRA_WHITE_SPACE_REGEX, " ", s.strip().replace("\n", " "))

base_gpt2_small = transformers.AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

tokenizer = transformers.AutoTokenizer.from_pretrained(
        "openai-community/gpt2",
        padding_side = "left",
        clean_up_tokenization_spaces = False)
tokenizer.pad_token = tokenizer.eos_token

prompt = "Write a poem:\n"
poem = ""
for i in range(STANZAS * LINES_PER_STANZA):
    line = generate(base_gpt2_small, prompt + poem, TOKENS_PER_LINE, args.temp)
    poem += normalize_whitespace(line) + "\n"
    if i % LINES_PER_STANZA == LINES_PER_STANZA - 1:
        poem += "\n"

print(poem)
