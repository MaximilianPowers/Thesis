import os
base = 'models'
sub_folders = os.listdir(base)
for folder in sub_folders:
    if len(folder.split('.')) > 1:
        continue
    models = os.listdir(os.path.join(base, folder))
    for model in models:
        if len(model.split('.')) > 1:
            continue
        os.mkdir(os.path.join(base, folder, model, 'logs'))
        open(os.path.join(base, folder, model, 'train.py'), 'x').close()
        open(os.path.join(base, folder, model, 'test.py'), 'x').close()