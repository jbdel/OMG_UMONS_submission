import csv
import os
import tensorflow as tf
import sys
import numpy as np
import subprocess


def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def write_ccc_csv(predictions, calculateEvaluationCCC, validationCSV, out_dir, verbose=False):

    with open(os.path.join(out_dir, 'out.csv'), 'w+') as csvfile:
        fieldnames = ['video', 'utterance', 'arousal', 'valence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(predictions):
            writer.writerow(
                {'video': 'None', 'utterance': 'None', 'arousal': predictions[i][0],
                 'valence': predictions[i][1]})

    proc = subprocess.Popen(
        ['python', calculateEvaluationCCC, validationCSV,
         os.path.join(out_dir, 'out.csv')],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    ret = (str(proc.communicate()[0]))
    if verbose:
        print(ret)
    arousal_ccc = float(find_between(ret, "Arousal CCC:  ", "\\n"))
    valence_ccc = float(find_between(ret, "Valence CCC:  ", "\\n"))
    mean_ccc = (arousal_ccc + valence_ccc) / 2
    print("Mean CCC:", mean_ccc, "Arousal CCC", arousal_ccc, "Valence CCC", valence_ccc)
    return mean_ccc, arousal_ccc, valence_ccc



def pad(seq, pad_length, num_features):
    diff = pad_length - len(seq)
    seq = np.array(seq)

    if diff ==0:
        return seq

    pad = np.zeros((diff, num_features), dtype=np.float32)
    seq = np.concatenate((seq,pad), axis=0)
    return seq


def get_sorted_data(data, mode, data_path):

    current_video = ""

    data_sorted = []
    data_video = []
    label_sorted = []
    label_video = []

    with open(os.path.join(data_path, "omg_{}Videos.csv".format(mode)), 'r') as csvfile:

        #checking if data size and csv size matches
        reader = csv.DictReader(csvfile)
        row_count = sum(1 for row in reader)
        assert (row_count == data.shape[0]), "Size mismatch for " + mode + ": "+ str(row_count) + " vs " + str(data.shape[0])
        csvfile.seek(0)

        #stacking video wise
        for i, row in enumerate(reader):
            if i == 0:
                continue

            if i == 1:
                current_video = row["video"]
                data_video.append(data[i-1])
                try:
                    label_video.append([row["arousal"], row["valence"]])
                except KeyError:
                    label_video.append([0.0, 0.0])
                continue

            if row["video"] != current_video:
                data_sorted.append(data_video)
                label_sorted.append(label_video)
                data_video = []
                label_video = []

                current_video = row["video"]

            data_video.append(data[i-1])
            try:
                label_video.append([row["arousal"], row["valence"]])
            except KeyError:
                label_video.append([0.0, 0.0])



        data_sorted.append(data_video)
        label_sorted.append(label_video)

    # all_length = sorted([len(x) for x in data_sorted])

    #get max length for padding
    max_length = max([len(x) for x in data_sorted])

    #saving real seq length (num utterance per video) for masking
    seq_length_sorted = [len(x) for x in data_sorted]

    #padding
    data_sorted = [pad(x,max_length, len(x[0])) for x in data_sorted]
    label_sorted = [pad(x,max_length, len(x[0])) for x in label_sorted]

    print("Mode ", mode," data shape:",np.array(data_sorted).shape)
    print("Mode ", mode," label shape:",np.array(label_sorted).shape)


    return np.array(data_sorted), np.array(label_sorted), np.array(seq_length_sorted), max_length



def batch_iter(X_sorted, Y_sorted, X_seq_length, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(X_sorted)
    num_batches_per_epoch = int(len(X_sorted)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data_X = X_sorted[shuffle_indices]
            shuffled_data_Y = Y_sorted[shuffle_indices]
            shuffled_seq_X  = X_seq_length[shuffle_indices]

        else:
            shuffled_data_X = X_sorted
            shuffled_data_Y = Y_sorted
            shuffled_seq_X  = X_seq_length
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if end_index-start_index<batch_size:
                start_index=start_index - (batch_size-(end_index-start_index))
            yield shuffled_data_X[start_index:end_index], shuffled_data_Y[start_index:end_index], shuffled_seq_X[start_index:end_index]

