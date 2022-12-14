# DeepGuessr


An AI for the web based game geoguessr. The model uses a ConVneXt-B backbone pretrained on ImageNet22k, which was then finetuned on a self collected dataset.


The conda environment required to train the model yourself can be created with the following cmd:
```
conda create -n <environment-name> --file req.txt
```

main thing is that it uses `pytorch 1.12.0` and `python 3.8.15`. Probably works with most environments, there isnt any special packages used.


For inference, I think it should be able to run with ~1GB of RAM on CPU. I haven't done any performance benchmarks so feel free to try it out and let me know. Training was done on a RTX 3090 ~30 epochs, note that I was able to train with `batch_size=7` with `torch.amp` which shortened training time from 1.5hrs per epoch to 50 minutes. 



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

# Results

## Histogram 
Naive: just predict x,y

Bin Based: predict state then residual 

- No pretrain is the bin based model without pretraining on Imagenet22k

random guess: guessing a random state 

always center: always predicting the center of the USA \[39.50,-98.35\]

![](graphs/zoomedin_dist.png)


## Mean distance from true location
![](graphs/Mean.png)

## Median distance from true location
![](graphs/Median.png)

## standard deviation of difference 
![](graphs/Standard%20Deviation.png)

# TODO:

Create browser extension to interface with game UI




