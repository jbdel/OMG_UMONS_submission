import os
import csv


ABS = "data"
for mode in ["Train", "Validation", "Test"]:
    _dict = {}
    #on ouvre d'abord les transcription et on fait un dictionnaire id -> transcription
    with open(os.path.join(ABS, "omg_{}Transcripts.csv".format(mode)), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            _dict[row['video'], row['utterance']] = row['transcript']

    #on ouvre ensuite le fichier des label, si l'id video pour le label existe ds le dico, ...
    # ... on prend le label, et on ecrit le transcript label ds train.txt
    with open(os.path.join(ABS, mode.lower()+".txt"), "w+") as writefile:
        with open(os.path.join(ABS, "omg_{}Videos.csv".format(mode)), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    transcript = _dict[row['video'], row['utterance']]
                except KeyError:
                    transcript = ""
                    print("Warning : ",mode,"video ",row['video'], row['utterance'],"has no stranscript")

                if len(row) == 5:
                    writefile.write(transcript+"\t"+"0"+"\t"+"0.0"+"\t"+"0.0"+"\t"+row['video']+"\t"+row['utterance']+"\n")
                else:
                    writefile.write(transcript+"\t"+row['EmotionMaxVote']+"\t"+row['arousal']+"\t"+row['valence']+"\t"+row['video']+"\t"+row['utterance']+"\n")

