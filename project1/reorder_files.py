import os
import shutil
import sys

if len(sys.argv) < 4:
    print('Usage:', sys.argv[0], 'LABEL_FILE DATA_ROOT DATA_DEST')
    exit(1)

with open(sys.argv[1], 'r') as labels_file:
    for line in labels_file:
        split = line.strip().split()
        gender, file_name = os.path.split(split[0])
        year = split[-1]
        src = os.path.join(sys.argv[2], split[0])
        directory = os.path.join(sys.argv[3], year)
        dest = os.path.join(directory, gender + file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        shutil.copyfile(src, dest)
