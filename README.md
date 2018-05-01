# OMG_UMONS_submission


### Requirements
* Python 3.4+
* Tensorflow 1.2+



### Monomodal feature extraction

Please refer to the text_cnn folder for linguistic features extraction.

Please refer to the video_model folder for visual features extraction.

Please refer to the audio_model folder for acoustic features extraction.

### Contextual monomodal feature extraction

Please refer to the context folder for features extraction.

### Contextual Multimodal feature extraction

Please refer to the context folder for features extraction.

### Scores


Results on dev set (averages on 10 runs)

| Modality  | CCC Arousal | CCC Valence | CCC Mean |
| ------------- | ------------- |------------- |------------- |
|  Monomodal feature extraction   |  |  | |
| Text - CNN   | 0.078  | 0.25 | 0.165  |
| Audio - OpenSmile Features | 0.045 | 0.21 | 0.15  |
| Video - 3DCNN   | 0.236  | 0.141 | 0.189 |
|  Contextual monomodal   |  | | |
| Text   |   | | 0.220  |
| Audio |  |  | 0.223 |
| Video  |   |  | 0.227 |
|  Contextual multimoal   |  | | |
A+T+V | 0.244   | 0.304 | 0.274 |
A+T+V+CBP | 0.280  | 0.321 | 0.301 |
