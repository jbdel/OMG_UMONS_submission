from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import numpy as np
import os
import csv

data_path = "data/"

x = np.load(os.path.join(data_path,"train.npy")).astype(np.float)
y = np.load(os.path.join(data_path,"valid.npy")).astype(np.float)
z = np.load(os.path.join(data_path,"test.npy")).astype(np.float)

print("Original feature size : ", x.shape[1])

#we load train target
utterances_x = []
with open(os.path.join(os.path.join(data_path,"omg_TrainVideos.csv")), 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        utterances_x.append([row['arousal'], row['valence']])



u = np.array(utterances_x).astype(np.float)

#we select best features according to both arousal and valence
s1 = SelectKBest(f_regression, k=80).fit(x, u[:,0])
s2 = SelectKBest(f_regression, k=80).fit(x, u[:,1])
r1 = s1.get_support(True)
r2 = s2.get_support(True)


#we merge best features
l = len(set(np.concatenate([r1,r2])))

print("feature size",l)

#for each dataset, we get the chosen features
train = np.zeros((x.shape[0],l))
valid = np.zeros((y.shape[0],l))
test = np.zeros((z.shape[0],l))

for i in range(x.shape[0]):
    for j,indice in enumerate(set(np.concatenate([r1,r2]))):
        train[i][j] = x[i][indice]

for i in range(y.shape[0]):
    for j,indice in enumerate(set(np.concatenate([r1,r2]))):
        valid[i][j] = y[i][indice]

for i in range(z.shape[0]):
    for j,indice in enumerate(set(np.concatenate([r1,r2]))):
        test[i][j] = z[i][indice]



np.save(os.path.join(data_path,"audio_train"), train)
np.save(os.path.join(data_path,"audio_validation"), valid)
np.save(os.path.join(data_path,"audio_test"), test)
