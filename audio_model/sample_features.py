from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import numpy as np
import os
import csv



x = np.load("IS13_ComParE_train.npy").astype(np.float)
y = np.load("IS13_ComParE_valid.npy").astype(np.float)

utterances_x = []
with open(os.path.join("omg_TrainVideos.csv"), 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        utterances_x.append([row['arousal'], row['valence']])



u = np.array(utterances_x).astype(np.float)


s1 = SelectKBest(f_regression, k=80).fit(x, u[:,0])
s2 = SelectKBest(f_regression, k=80).fit(x, u[:,1])
r1 = s1.get_support(True)
r2 = s2.get_support(True)


l = len(set(np.concatenate([r1,r2])))

print("feature size",l)

train = np.zeros((x.shape[0],l))
valid = np.zeros((y.shape[0],l))

for i in range(x.shape[0]):
    for j,indice in enumerate(set(np.concatenate([r1,r2]))):
        train[i][j] = x[i][indice]

for i in range(y.shape[0]):
    for j,indice in enumerate(set(np.concatenate([r1,r2]))):
        valid[i][j] = y[i][indice]


np.save("audio_train", train)
np.save("audio_validation", valid)
