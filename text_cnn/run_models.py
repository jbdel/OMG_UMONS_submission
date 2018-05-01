import os
import sys
from subprocess import Popen

# run all child scripts in parallel
# processes = [Popen([sys.executable, filename], cwd=dirpath)
#              for dirpath, dirname , filenames in os.walk('.')
#              for filename in filenames
#              if filename == 'Test.py']
#
# # wait until they finish
# for p in processes:
#     p.wait()
# print("all done")
processes = ["train.py" for _ in range(10)]


for n in processes:
    p = Popen([sys.executable, n], cwd=os.getcwd())
    p.wait()
print("all done")