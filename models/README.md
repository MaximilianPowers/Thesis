## MODEL SPECS

Models are separated into two types: unsupervised vs. supervised. Within each of these two folders, there are various PyTorch models, such as classical multi-layered perceptrons, a CNN etc. The only abstract constraint placed on models is that their layers, `model.layer` , is stored as a `nn.ModuleList` so that their forward passes can be saved when `save_activations=True`. 

For each model folder we have the following:

- /figures
- /logs
- /saved_models
- `log_ref.json`
- `model.py`
- `notebook.ipynb`
- `train.py`
- `test.py`

Everytime train.py is called, along with a step of parameters parsed by argparser, a new entry is added to log_ref.json which stores the values of the parameters in a dictionary whose key increments automatically with every call. 

test.py is used to compare various models against defined metrics, such as when performing hyperparameter optimisation.

Using the KEY of log_ref as an index, that model can then be retrieved in `/saved_models/KEY/model_{epoch}.pt`, any related figures, such as decision boundaries, can be found in `/figures/KEY/{epoch}.png`, and logging values are output to `/logs/KEY.csv`.

The jupyter notebooks contain miscellaneous code, such as post training evaluation and interpretability of models, ensuring metrics are working correctly, or overall development of new code.