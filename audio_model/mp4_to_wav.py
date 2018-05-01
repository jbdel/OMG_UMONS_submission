import os
import fnmatch
import subprocess
import sys
import csv

if not os.path.exists("data"):
    os.makedirs("data")

matches = []
for root, dirnames, filenames in os.walk('/media/jb/DATA/OMGEmotionChallenge/train'):
    for filename in fnmatch.filter(filenames, '*.mp4'):
        matches.append(os.path.join(root, filename))


filter = [x for x in matches if "temp" not in x]
print(len(filter),"files found")
for mode in ["Train","Validation"]:

    ABS = "/media/jb/DATA/OMGEmotionChallenge/"

    with open(ABS+"omg_{}Videos.csv".format(mode), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            found = False
            for x in filter:
                if (row['video'] in x) and (row['utterance'] in x):
                    blocks = x.split("/")
                    outfile = os.path.join("data","{}#{}".format(blocks[-2], blocks[-1]).replace("mp4", "wav"))
                    if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
                        print(outfile, "already exists")
                    else:
                        p = subprocess.call(["ffmpeg",
                                             "-y",
                                             "-i", x,
                                             "-ac", "1",
                                             "-f", "wav",
                                             "-ar", "16000",
                                             outfile],
                                            )
                    found=True
                    break

            if found == False:
                print("Warning : mp4 not found :", row['video'], row['utterance'])
                sys.exit(1)




