# triplet-ReID-pytorch
This is a simple implementation of the algorithm proposed in paper [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737).

This project is based on pytorch0.4.0 and python3. 

To be straight-forward and simple, only the method of training on pretrained Resnet-50 with batch-hard sampler is implemented.


To train on the Market1501 dataset, just run the train script:  
```
    $ python3 train.py
```
This will train an embedder model based on ResNet-50. The trained model will be stored in the path of ```/res/model.pkl```.


To embed the gallery set and query set of Market1501 run the embed script:
```
    $ python3 embed.py \
      --dataset_mode gallery \
      --store_pth ./res/emb_gallery.pkl \
      --data_pth datasets/Market-1501-v15.09.15/bounding_box_test

    $ python3 embed.py \
      --dataset_mode query \
      --store_pth ./res/emb_query.pkl \
      --data_pth datasets/Market-1501-v15.09.15/query
```
This script will use the trained embedder to embed the gallery and query set of Market1501, and store the embeddings as ```/res/embd_gallery.pkl``` and ```/res/emb_query.pkl```.
