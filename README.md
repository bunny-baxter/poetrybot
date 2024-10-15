# poetrybot

An exploration in fine-tuning gpt-2 to write poetry, using public domain poets.

## Scraper

`scraper.py` is a script that downloads all the poems by a particular author from public-domain-poetry.com. The site's robot.txt file politely asks for 10 seconds between requests from crawlers, so the script will sleep after every download. The poems are saved to download_cache/ in the current directory, and each poem gets its own file with the title, author, and full text.

Example usage:
```
python3 scraper.py http://public-domain-poetry.com/emily-elizabeth-dickinson
```

Thanks to the site maintainer for hosting so much poetry, though unfortunately I could not find their name or any way to contact them on the site.

## Poem generation

`generate-poem.py` is a script that generates a poem from a language model. The poem will be exactly 256 tokens split into 16 lines over 4 stanzas. Any newlines in the model output are replaced by spaces, to preserve the stanzas. By default it downloads [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) (which is the small version of the model) from huggingface and uses that. Use `--model` to specify a .pt file to load instead. Use `--temp` and `--topp` to specify temperature and nucleus sampling probability respectively (both default to 1.0).

## Training

`train.py` is a script that takes a copy of gpt-2 (same model as previous section) and trains it. Hyperparameters are defined as constants inside the script. To specify the dataset, pass in a .json file with directories and other .json files to load poem text from. Directories are assumed to be in the format that `scraper.py` generates. Other .json files are assumed to be in the format of [this file](https://cummings.ee/downloads/poems.json), though the script only reads the "text" field of each element.

Example config file:
```
{
  "directory": [  "emily-elizabeth-dickinson" ],
  "json": [ "poems.json" ]
}
```

Example script usage:
```
python3 train.py download_cache/config.json --device cuda
```

When finished, the resulting model is saved as a .pt file to checkpoints/ in the current directory.
