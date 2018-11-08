# triplet-ReID-pytorch
This is a simple implementation of the algorithm proposed in paper [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737).

This project is based on pytorch0.4.0 and python3. 

To be straight-forward and simple, only the method of training on pretrained Resnet-50 with batch-hard sampler(*TriNet* according to the authors) is implemented.


### prepare dataset
Run the script of ```datasets/download_market1501.sh``` to download and uncompress the Market1501 dataset.
```
    $ cd triplet-reid-pytorch/datasets
    $ sh download_market1501.sh 
```

### train the model
* To train on the Market1501 dataset, just run the training script:  
```
    $ cd triplet-reid-pytorch
    $ python3 train.py
```
This will train an embedder model based on ResNet-50. The trained model will be stored in the path of ```/res/model.pkl```.


### embed the query and gallery dataset
* To embed the gallery set and query set of Market1501, run the corresponding embedding scripts:
```
    $ python3 embed.py \
      --store_pth ./res/emb_gallery.pkl \
      --data_pth datasets/Market-1501-v15.09.15/bounding_box_test

    $ python3 embed.py \
      --store_pth ./res/emb_query.pkl \
      --data_pth datasets/Market-1501-v15.09.15/query
```
These scripts will use the trained embedder to embed the gallery and query set of Market1501, and store the embeddings as ```/res/embd_gallery.pkl``` and ```/res/emb_query.pkl```.


### evaluate the embeddings
* Then compute the rank-1 cmc and mAP:  
```
    $ python3 eval.py --gallery_embs ./res/emb_gallery.pkl \
      --query_embs ./res/emb_query.pkl \
      --cmc_rank 1
```
This will evaluate the model with the query and gallery dataset.


### Notes
After refering to some other paper and implementations, I got to to know two tricks that help to boost the performance:   
* adjust the stride of the last stage of resnet from 2 to 1.
* use augmentation method of [random erasing](https://arxiv.org/abs/1708.04896).

With these two tricks, the mAP and rank-1 cmc on Market1501 dataset reaches 76.04/88.27, much higher than the result claimed in the paper.
