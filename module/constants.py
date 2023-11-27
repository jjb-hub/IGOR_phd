######### HERE WE DECLARE THE CONSTANTS USED BY OTHER FILES ############
# Constant are meant to be constants, the should not changed, that's what variables or user are for

import os

### Constants that reflect the filesystem structure, used by util functions
ROOT = os.getcwd()  # This gives terminal location (terminal working dir)
INPUT_DIR = f"{ROOT}/input"
OUTPUT_DIR = f"{ROOT}/output"
CACHE_DIR = f"{INPUT_DIR}/cache"