# Attention pytorch implementation on FMRI data
VoxResNet with DeepExplain
Psycho deviation in behaviour
https://openneuro.org/datasets/ds000030

# Description
 
- `tensorflow_train.ipynb`

  We define VoxResNet model, train it and save to file system.
- `tensorflow_explain.ipynb`

  Pytorch implementation can be found in:
- `main_pytorch.ipynb`
  We use DeepExplain framework to decribe predictions of trained model

To run these files you need to import them to Colab and unpack `sMRI` dataset to folder `Colab Notebooks/sMRI` at your google drive.


If notebook fails to load in github you can use nbviewer to see notebooks correctly
- [tensorflow_train.ipynb](https://nbviewer.jupyter.org/github/stepankonev/attention_project/blob/master/tensorflow_train.ipynb)
- [tensorflow_explain.ipynb](https://nbviewer.jupyter.org/github/stepankonev/attention_project/blob/master/tensorflow_explain.ipynb)
