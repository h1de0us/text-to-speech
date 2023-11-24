## Installation guide

To install all the requirements, run
```shell
pip install -r ./requirements.txt
```
To download and preprocess the data, run the following:
```shell
bash run.sh download_data
```
To train a model (default Fast Speech), run
```shell
cd text-to-speech
python3 train.py -c hw_tts/configs/fast_speech_config.json
```

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.


Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize