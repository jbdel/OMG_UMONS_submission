import os
import sys
from subprocess import Popen
import argparse

#
# python run_models.py \
#             --data_path ./data/ \
#             --train audio_train.npy \
#             --validation audio_validation.npy \
#             --stack_num 3 \
#             --d_k 64 \
#             --d_v 64 \
#             --h 10  \
#             --d_ff 1024  \
#             --dropout_keep 0.8

# python run_models.py \
#             --data_path ./data/ \
#             --train video_train.npy \
#             --validation video_validation.npy \
#             --stack_num 3 \
#             --d_k 64 \
#             --d_v 64 \
#             --h 10  \
#             --d_ff 1024  \
#             --dropout_keep 0.8


# python run_models.py \
#             --data_path ./data/ \
#             --train train.txt.npy \
#             --validation validation.txt.npy \
#             --stack_num 3 \
#             --d_k 64 \
#             --d_v 64 \
#             --h 10  \
#             --d_ff 1024 \
#             --dropout_keep 0.8


# python run_models.py \
#             --data_path ./data/ \
#             --train text_train.npy \
#             --validation text_validation.npy \
#             --stack_num 3 \
#             --d_k 64 \
#             --d_v 64 \
#             --h 10  \
#             --d_ff 1024 \
#             --dropout_keep 0.8


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="./data/")
parser.add_argument('--train',     type=str, default="train.txt.npy")
parser.add_argument('--validation',type=str, default="validation.txt.npy")
parser.add_argument('--stack_num', type=int, default=2)
parser.add_argument('--d_k',       type=int, default=64)
parser.add_argument('--d_v',       type=int, default=64)
parser.add_argument('--h',         type=int, default=8)
parser.add_argument('--d_ff',         type=int, default=1024)
parser.add_argument('--dropout_keep', type=float, default=0.8)

args = parser.parse_args()

for _ in range(10):
    p = Popen(["python", "main.py",
               "--data_path="+args.data_path,
               "--train="+args.train,
               "--validation="+args.validation,
               "--stack_num="+str(args.stack_num),
               "--d_k="+str(args.d_k),
               "--d_v="+str(args.d_v),
               "--h="+str(args.h),
               "--d_ff="+str(args.d_ff),
               "--dropout_keep="+str(args.dropout_keep),
               ], cwd=os.getcwd())
    p.wait()
print("all done")