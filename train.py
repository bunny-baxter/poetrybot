# Trains the poetry model.

import argparse
import collections
import copy
import datetime
import os
import json
import time
import torch
import transformers

parser = argparse.ArgumentParser(prog = "Train poetry model")
parser.add_argument("dataset_config", help = "Json file listing which files and directories to load poems from")
args = parser.parse_args()

base_model = transformers.AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

tokenizer = transformers.AutoTokenizer.from_pretrained(
        "openai-community/gpt2",
        padding_side = "left",
        clean_up_tokenization_spaces = False)
tokenizer.pad_token = tokenizer.eos_token

TOKEN_COUNT = len(tokenizer)

def load_poems_from_directory(dirpath):
    poems = []
    for filename in os.listdir(dirpath):
        if filename == "author_index.html":
            continue
        filepath = os.path.join(dirpath, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            contents = f.read()
        within_poem = False
        poem = ""
        for line in contents.splitlines():
            if within_poem:
                if line.strip() == "*** END TEXT ***":
                    break
                poem += line + "\n"
            else:
                if line.strip() == "*** START TEXT ***":
                    within_poem = True
        if poem:
            poems.append(poem)
    return poems

def load_poems_from_json(json_filepath):
    poems = []
    with open(json_filepath, "r", encoding = "utf-8") as f:
        json_contents = json.loads(f.read())
        for entry in json_contents:
            poems.append(entry["text"])
    return poems

class PrefixNextTokenDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, tokenizer, prefix_max_size, transform = None, target_transform = None):
        self.transform = transform
        self.target_transform = target_transform

        self.items = []
        for text in text_list:
            tokenizer_result = tokenizer(text, padding = True)
            tokenized_text = tokenizer_result.input_ids
            token_window = collections.deque()
            for i in range(len(tokenized_text) - 1):
                token_window.append(tokenized_text[i])
                if len(token_window) == prefix_max_size:
                    self.items.append((list(token_window), tokenized_text[i + 1]))
                    token_window.popleft()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        prefix, next_token = self.items[i]
        if (self.transform):
            prefix = self.transform(prefix)
        if (self.target_transform):
            next_token = self.target_transform(next_token)
        return (torch.tensor(prefix), torch.nn.functional.one_hot(torch.tensor(next_token).to(torch.int64), TOKEN_COUNT))

all_poems = []

dataset_config_filepath = args.dataset_config
dataset_config_basedir = os.path.dirname(dataset_config_filepath)
with open(dataset_config_filepath, "r", encoding = "utf-8") as f:
    json_contents = json.loads(f.read())
    if "json" in json_contents:
        for filename in json_contents["json"]:
            all_poems += load_poems_from_json(os.path.join(dataset_config_basedir, filename))
    if "directory" in json_contents:
        for dirname in json_contents["directory"]:
            all_poems += load_poems_from_directory(os.path.join(dataset_config_basedir, dirname))

assert len(all_poems) > 0

TOKEN_WINDOW_SIZE = 32
dataset = PrefixNextTokenDataset(all_poems, tokenizer, TOKEN_WINDOW_SIZE)

training_model = copy.deepcopy(base_model)

def train(dataloader, model, loss_fn, optimizer):
    batch_count = len(dataloader)
    model.train()
    for batch_index, (prefixes, next_tokens_expected) in enumerate(dataloader):
        output = model(prefixes)
        next_token_probabilities = output.logits.softmax(dim = 2)
        loss = loss_fn(next_token_probabilities, next_tokens_expected)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"finished batch {batch_index+1}/{batch_count}: loss = {loss.item()}")

LEARNING_RATE = 0.00001
BATCH_SIZE = 256
EPOCHS = 1

dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(training_model.parameters(), lr = LEARNING_RATE)

begin_time = time.time()
for epoch in range(EPOCHS):
    print(f"beginning epoch {epoch+1}/{EPOCHS}...")
    train(dataloader, training_model, loss_fn, optimizer)
    print()
end_time = time.time()
print(f"training time was {end_time - begin_time} seconds")

date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
torch.save(training_model, f"checkpoints/poetrybot {date_str}.pt")
