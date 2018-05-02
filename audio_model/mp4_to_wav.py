import os
import fnmatch
import subprocess
import sys
import csv
import argparse




parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, default="/media/jb/DATA/OMGEmotionChallenge/train")
parser.add_argument('--out_dir', type=str, default="./wav/")
args = parser.parse_args()


data_dir = "./data/"

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)


matches = []
for root, dirnames, filenames in os.walk(args.video_path):
    for filename in fnmatch.filter(filenames, '*.mp4'):
        matches.append(os.path.join(root, filename))


filter = [x for x in matches if "temp" not in x]
print(len(filter),"files found")
for mode in ["Train","Validation", "Test"]:


    with open(data_dir+"omg_{}Videos.csv".format(mode), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            found = False
            for x in filter:
                if (row['video'] in x) and (row['utterance'] in x):
                    blocks = x.split("/")
                    outfile = os.path.join(args.out_dir,"{}#{}".format(blocks[-2], blocks[-1]).replace("mp4", "wav"))
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




