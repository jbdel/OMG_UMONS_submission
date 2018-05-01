import keras
import os
import csv
import subprocess
import numpy as np
from tqdm import tqdm
import sys
import collections


def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def loaddata(video_dir, vid3d, args, mode, color=False, skip=True):
    #if validation, we need to create sorted dataset them accordingly
    files = []
    csv_file = args.trainCSV if mode == "Train" else args.validationCSV
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            filename = row["video"] + "#" +row["utterance"]
            files.append([filename, row["arousal"], row["valence"]])
    X = []
    labels = []

    pbar = tqdm(total=len(files))

    for i,f in enumerate(files):
        filename, arousal, valence = f
        pbar.update(1)
        name = os.path.join(video_dir, filename)
        label = [float(arousal), float(valence)]
        labels.append(label)
        X.append(vid3d.video3d(name, color=color, skip=skip))

    pbar.close()

    if color:
        return np.array(X).transpose((0, 2, 3, 4, 1)), labels
    else:
        return np.array(X).transpose((0, 2, 3, 1)), labels


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
    return mean_ccc,  arousal_ccc, valence_ccc



class prediction_history(keras.callbacks.Callback):
    def __init__(self, X_test, model, args, batch_size=8):

        self.x_test = X_test
        self.model = model
        self.batch_size = batch_size
        self.args = args
        self.ccc = {}
        self.best_ccc = [0.0,0.0,0.0]
        self.early_stop = 0

    def write_scores(self):
        od = collections.OrderedDict(sorted(self.ccc.items()))
        with open(os.path.join(self.args.out_dir,"best_scores.txt"), "a+") as f:
            for key, value in od.items():
                f.write("{} \t {} \n".format(key, value))

        with open(os.path.join("runs","best_scores_overall.txt"), "a+") as f:
            f.write("{}({}/{}) \t {} \n".format(self.best_ccc[0], self.best_ccc[1], self.best_ccc[2], self.args.out_dir))

    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(self.x_test, batch_size=self.batch_size)
        mean_ccc = write_ccc_csv(predictions, self.args.calculateEvaluationCCC, self.args.validationCSV, self.args.out_dir, verbose=True)

        if mean_ccc[0] > self.best_ccc[0]:
            self.best_ccc = mean_ccc
            self.ccc[self.best_ccc[0]] = epoch+1
            self.early_stop = 0

        if self.early_stop == 10:
            self.write_scores()
            sys.exit()

        self.early_stop +=1
        print("early_stop",self.early_stop)


    def on_train_end(self, logs=None):
        self.write_scores()

