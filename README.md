# Zalando's MNIST fashion replacement

Zalando recently released an MNIST replacement. The issues with using MNIST are
known but you can read about the dataset and their motivation [here](https://github.com/zalandoresearch/fashion-mnist).

### Training
```
python train.py

        --model         # specify model, (FashionSimpleNet, resnet18)
        --patience      # early stopping
        --batch_size    # batch size
        --nepochs       # max epochs
        --nocuda        # no cuda
        --nworkers      # number of workers
        --seed          # random seed
        --data          # mnist, fasion
```

#### Preprocessing

Normalization, random horizontal flip, random vertical flip, random translation, random rotation.

### Results
Best accuracy of the same model when run on MNIST and Fashion MNIST.

|Model|MNIST|Fashion MNIST|
|---|---|---|
|ResNet18| 0.995 | 0.949|
|SimpleNet|0.994|0.919|
