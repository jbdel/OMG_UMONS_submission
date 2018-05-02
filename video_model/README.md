Requirements:
Keras 2.1.6 
opencv-python 3.4.0.12   
tqdm 4.23.1     

This module use 3D-convolution to extract features, as seen in Shuiwang Ji, Wei Xu, Ming Yang, and Kai Yu work :  [3d convolutional neural networks for human action recognition](https://ieeexplore.ieee.org/abstract/document/6165309/):  

Firstly, we create three folders : [Train,Validation,Test]_videos, where each utterances is of the form 
videoid#utterance.mp4. (We use symbolic links, we dont duplicate the videos)

As arugment, the script takes the folder where videos are stored (created from preprocess.py provided by challenge author)
If a video of the csv's is not found, script raisse an error.

```bash
python make_dataset.py --video_path /.../videos/
```

To start training, use :

```bash
python 3dcnn.py
```

Before training, the scripts will create dataset according to the specifications (32 frames of size 32x32 by default)

Model stops training if ccc score didnt improve for 10 epochs.

To extract features, we use :

```bash
python eval.py --npz_file ./data/dataset_[Train, Validation]_2_32_True.npz \
    --fileCSV ./data/omg_TrainVideos.csv \
    --out_name video_[train, validation] \
    --model_dir ./runs/xxxxxx \
    --weights_file xxxxx.hdf5
    
python eval.py --npz_file ./data/dataset_Test_2_32_True.npz \
    --out_name video_test \
    --model_dir ./runs/xxxxxx \
    --weights_file xxxxx.hdf5 \
    --eval_ccc False
```

To runs 10 models, use:
```bash
python run_models.py
```

to get a mean evaluation prediction score over ten runs
