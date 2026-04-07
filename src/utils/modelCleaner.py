# Used to reset models to their very first state

import os
import glob

files = glob.glob("./claris*.pth")

for f in files: 
    try:
        os.remove(f)
        print(f"Removed {f}")
    except OSError as e:
        print(f"Error while deleting {f} : returned {e.strerror}")
