Requirements: 
gensim-3.4.0 


Linguistic features are extracted by a CNN model

Reference : Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper.

First, create data set from transcript, output goes into data folder

```bash
python preprocess.py
```

Now, we need to download word2vec embeddings.

Here is a mirroring link of the data from the official [word2vec website](https://code.google.com/archive/p/word2vec/):  
[GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) and place it in the data
folder.

We can start a training :

```bash
    python train.py \
        --train_data_path ./data/train.txt \
        --dev_data_path ./data/validation.txt \
        --use_word2vec True \
        --word2vec ./data/GoogleNews-vectors-negative300.bin \
        --w2v_data_path ./data/w2v.npy \
        --filter_sizes 3,4,2 \
        --num_filters  30,30,60 \
        --dropout_keep_prob 0.8 \
        --batch_size 8
```

Checkpoints and scores are stored in runs 'runs' folder.

Alternatively, you can fill these parameters in train.py and start 10 trainings by running

```bash
python run_models.py
```
to get a mean performance over 10 runs.

After training, you can pick a model and extract the features per utterance :

```bash
python eval.py \
    --test_data_path ./data/train.txt \
    --validationCSV ./data/omg_TrainVideos.csv


python eval.py \
    --test_data_path ./data/validation.txt \
    --validationCSV ./data/omg_ValidationVideos.csv


python eval.py \
    --test_data_path ./data/test.txt \
    --compute_ccc False
```
