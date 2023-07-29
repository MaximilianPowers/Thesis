import os
base = 'models'

for mode in ['supervised', 'unsupervised']:
    sub_folders = os.listdir(os.path.join(base, mode))

    for folder in sub_folders:
        if os.path.isdir(os.path.join(base, mode, folder)):
            open(os.path.join(base, mode, folder, 'log_ref.json'), 'x').close()