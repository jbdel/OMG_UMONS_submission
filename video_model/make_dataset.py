import os
import fnmatch
import sys
import csv

ABS = "/media/jb/DATA/OMGEmotionChallenge/"




matches = []
for root, dirnames, filenames in os.walk(ABS+'train'):
    for filename in fnmatch.filter(filenames, '*.mp4'):
        matches.append(os.path.join(root, filename))

filter = [x for x in matches if "temp" not in x]

for mode in ["Train","Validation"]:

    data_dir = "{}_videos".format(mode)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    with open(ABS + "omg_{}Videos.csv".format(mode), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            found = False
            for x in filter:
                if (row['video'] in x) and (row['utterance'] in x):
                    try:
                        filename = os.path.join(data_dir,row['video'] +"#"+row['utterance'])
                        os.symlink(x, filename)
                    except FileExistsError:
                        print(os.path.join(data_dir,filename,"exists already"))
                    found = True
                    break
            if found == False:
                print("Warning : mp4 not found :", row['video'], row['utterance'])
                sys.exit(1)

