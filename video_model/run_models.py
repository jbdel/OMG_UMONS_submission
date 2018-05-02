import os
import sys
from subprocess import Popen

processes = ["3dcnn.py" for _ in range(10)]

for n in processes:
    p = Popen([sys.executable, n], cwd=os.getcwd())
    p.wait()
print("all done")