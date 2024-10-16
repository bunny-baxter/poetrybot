# Generates a poem.

import argparse
import torch
import transformers

def validate_topp(value):
    value = float(value)
    if value <= 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("Nucleus sampling probability must be in (0, 1].")
    return value

parser = argparse.ArgumentParser(prog = "Generate Poem")
parser.add_argument("--temp", type = float, default = 1.0, help = "model temperature")
parser.add_argument("--topp", type = validate_topp, default = 1.0, help = "nucleus sampling cumulative probability")
parser.add_argument("--model", default = None, help = "model file to load")
parser.add_argument("--device", choices = ["cpu", "cuda", "mps"], default = "cpu", help = "Tensor device")
args = parser.parse_args()

TOKENS_PER_LINE = 16
LINES_PER_STANZA = 4
STANZAS = 4

def _infer_once(model, text_input, temperature, nucleus_probability):
    with torch.no_grad():
        tokenized = tokenizer(text_input, return_tensors = "pt")
        # TODO: Re-tokenizing the input on every new token is probably slowing generation down a lot.
        input_ids = tokenized.input_ids.to(args.device)
        output = model(input_ids, attention_mask = tokenized.attention_mask)
        next_token_probabilities = output.logits[0][-1]
        if temperature == 0.0:
            next_token = next_token_probabilities.argmax(0)
        else:
            if temperature != 1.0:
                next_token_probabilities = next_token_probabilities / temperature
            next_token_probabilities = next_token_probabilities.softmax(0)
            if nucleus_probability < 1.0:
                sorted_probabilities, sorted_indexes = torch.sort(next_token_probabilities, descending = True)
                i = 0
                while i < len(sorted_probabilities) and sorted_probabilities[i] <= nucleus_probability:
                    nucleus_probability -= sorted_probabilities[i]
                    i += 1
                if i == 0 or i == 1:
                    # Either the most likely token is more likely than `nucleus_probability`, or it
                    # plus the second most likely token are together more likely. Either way, just
                    # take the most likely token.
                    next_token = sorted_indexes[0]
                else:
                    top_probabilities = sorted_probabilities[:i]
                    next_token = sorted_indexes[torch.multinomial(top_probabilities, 1)[0]]
            else:
                next_token = torch.multinomial(next_token_probabilities, 1)[0]
        return tokenizer.decode(next_token)

def generate(model, prompt, n_tokens, temperature, nucleus_probability):
    output = ""
    for _ in range(n_tokens):
        next_token = _infer_once(model, prompt + output, temperature, nucleus_probability)
        if next_token == "<|endoftext|>":
            break
        output += next_token
    return output

def remove_newlines(s):
    return s.strip().replace("\n", " ")


if args.model:
    model = torch.load(args.model).to(args.device)
else:
    model = transformers.AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(args.device)

tokenizer = transformers.AutoTokenizer.from_pretrained(
        "openai-community/gpt2",
        padding_side = "left",
        clean_up_tokenization_spaces = False)
tokenizer.pad_token = tokenizer.eos_token

prompt = "Write a poem:\n"
poem = ""
for i in range(STANZAS * LINES_PER_STANZA):
    line = generate(model, prompt + poem, TOKENS_PER_LINE, args.temp, args.topp)
    line = remove_newlines(line)
    if i % LINES_PER_STANZA == LINES_PER_STANZA - 1:
        line += "\n"
    poem += line
    print(line)
