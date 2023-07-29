import os
base = 'metrics'
sub_folders = os.listdir(base)

for folder in sub_folders:
    if os.path.isdir(os.path.join(base, folder)):
        os.mkdir(os.path.join(base, folder, 'utils'))