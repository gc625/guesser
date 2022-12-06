# guesser.ai


An AI for the web based game geoguessr. The model uses a ConVneXt-B backbone pretrained on ImageNet22k, which was then finetuned on the dataset.


The conda environment required to train the model yourself can be created with the following cmd:
```
conda create -n <environment-name> --file req.txt
```

main thing is that its `pytorch 1.12.0` and python 3.8.15


# Downloading dataset

The data used to train the bot is pulled from Google street view static API. You will first need to create an account and fill in your API key and secret SHA256 hash in `creds.py`. Then run 

```
create_data.py
```

The model then can be trained by running `main.py` with your own args.

# Model Checkpoints

The weights to the two models can be found here: [drive link](https://drive.google.com/drive/folders/1D2474a_rjvMkjYEUSQY1yzG4DbLVRFx9?usp=share_link)


# Current capabilities
- currently *only* works for USA. I'm too poor to download enough training images for the entire world. Paypal me $20k and I'll do it

# TODO:

Create browser extension to interface with game UI




