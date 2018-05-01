import argparse
import os
import sys


import numpy as np
from keras.models import Sequential, model_from_json
from keras import backend as K
import utils

# python eval.py --npz_file ./data/dataset_Validation_2_32_True.npz \
#     --fileCSV ./data/omg_ValidationVideos.csv \
#     --out_name video_validation \
#     --model_dir ./runs/1525120165 \
#     --weights_file weights-improvement-22-0.08.hdf5


# python eval.py --npz_file ./data/dataset_Train_2_32_True.npz \
#     --fileCSV ./data/omg_TrainVideos.csv \
#     --out_name video_train \
#     --model_dir ./runs/1525120165 \
#     --weights_file weights-improvement-22-0.08.hdf5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_file', type=str, default="./data/dataset_Validation_2_32_True.npz")
    parser.add_argument('--model_dir', '-m', type=str, default="./runs/1525120165",
                        required=False, help='json of model shape')
    parser.add_argument('--weights_file', '-w', type=str, default="weights-improvement-22-0.08.hdf5",
                        required=False, help='hd5 of weights')
    parser.add_argument('--out_name', type=str, default="validation")
    parser.add_argument('--calculateEvaluationCCC', type=str, default="./data/calculateEvaluationCCC.py")
    parser.add_argument('--fileCSV', type=str, default="./data/omg_ValidationVideos.csv")


    args = parser.parse_args()


    # Loading data
    loadeddata = np.load(args.npz_file)
    X, Y = loadeddata["X"], loadeddata["Y"]
    print('Data : X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))

    #Loading model params
    json_file = os.path.join(args.model_dir, "model.json")
    with open(json_file, 'r') as model_json:
        model = model_from_json(model_json.read())

    # Loading weights
    weights_file = os.path.join(args.model_dir, args.weights_file)
    model.load_weights(weights_file)

    #Specifying inputs and outputs
    get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.layers[9].output, model.layers[11].output])


    layer_outputs = []
    predictions = []
    #we run model batch by batch
    for i,x in enumerate(X):
        #utterance x is of size (W,D,T), need to reshape to (1,W,D,T)
        x = np.expand_dims(x, axis=0)

        # (0 means test phase)
        layer_output,prediction = get_3rd_layer_output([x, 0])

        # (dim 0 to avoid useless (1, example))
        layer_outputs.append(layer_output[0])
        predictions.append(prediction[0])

    print("Features shape:",np.array(layer_outputs).shape)
    print("Predictions shape:",np.array(predictions).shape)
    feature_file = os.path.join(args.model_dir, args.out_name)
    np.save(feature_file, np.array(layer_outputs))
    print("Saving features at ", feature_file+".npy")
    utils.write_ccc_csv(np.array(predictions), args.calculateEvaluationCCC, args.fileCSV, args.model_dir, verbose=False)


if __name__ == '__main__':
    main()
