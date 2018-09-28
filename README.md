# triplet-network-pytorch
This is a simple implementation of the algorithm proposed in paper [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737).

This project is based on pytorch0.4.0 and python3. 

To be straight-forward and simple, only the method of training on pretrained Resnet50 with batch-hard sampler is implemented.


To train on the Market1501 dataset, just run the train script:  
```
    $ python3 train.py
```
This will train an embedder model based on ResNet-50. The trained model will be stored in the directory of ```/res```.


To embed the gallery set of Market1501 run the embed script:
```
    $ python3 embed.py
```
This script will use the trained embedder to embed the gallery set and store the embeddings as ```/res/embds.pkl```.
