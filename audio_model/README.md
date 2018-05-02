Requirements:

scikit-learn (0.19.1)

This module extract acoustic features for each utterances with opensmile (IS13_ComParE config).

First, install opensmile stand-alone :

```bash
  unpack the archive:
    tar-zxvf opensmile-X.X.X.tar.gz
    cd opensmile-X.X.X
    (replace X.X.X by the version number of your version)

  then type:
    sh buildStandalone.sh
```

Now, lets extract sound from mp4 files. As arugment, the script takes the folder where videos are stored (created from preprocess.py provided by challenge author) If a video of the csv's is not found, script raisse an error.

```
python make_dataset.py --video_path /.../videos/
```

Wav files are created in wav folder. Now, we need to get features, we launch :

```bash
python extract.py --open_smile_path xxxx/opensmile-2.3.0/
```

Three files, [train, valid, test].npy, are created in the data folder containing 6373 features for each utterances.

We then extract the 80 most usefull features for both arousal and valence according to the training set. To do so, we use the ```SelectKBest``` function from
**sk-learn**. We merge the both best selection of features and end up with 121 features.

```
python sample_features.py
```

New sampled feature files, audio_[train,validation,test].npy, are stored in data folder.


    
    
