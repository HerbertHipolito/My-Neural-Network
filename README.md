# Artificial Neural Network (ANN)
A simple implementation of a Multi-Layer Perceptron (MLP). Additional features were also implemented such as momentum, L1 and L2 regularization, and Early stopping.
# GridSearch
In this project, I still used some libraries to help me as Numpy, Matplotlib, and Sklearn for the GridSearch, metrics (accuracy, precision, recall, and F1), and comparison of results. To measure my model performance, I carry out a small set of simulations with different hyperparameters to achieve the best results possible. The same tests were also performed in Sklearn ANN in order to compare both models and figure out how good my model was. The hyperparameters utilized are described in "parameters.py" => hyperparameter dict variable.
#  Results
Unfortunately, the comparisons are not totally fair because I did not implement all parameters used in the Sklearn ANN and the ANN contains a lot of internal randomness, for example, the way that the initial weights are initialized and data shuffling during at each epoch. However, my ANN had 2% metric results below the Sklearn ANN on average. The following graphs help us to view it.

## 20 neurons, 2 hidden leayer, momentum  0.4, learning Rate = 0.001

![plot_4](https://user-images.githubusercontent.com/94997683/236550267-551b36e9-defb-4f51-8190-02817ca1e13f.png)

## 25 neurons, 2 hidden leayer, momentum  0, learning Rate = 0.001

![plot_2](https://user-images.githubusercontent.com/94997683/236551117-150b9f81-34ac-4624-9fdf-b0b696b7495b.png)

# Results in test dataset

The dataset was generated from sklearn.datasets.make_classification to train and test the models. 
It was selected 5 features (x) and 2 classes (y) with 7500 samples.
