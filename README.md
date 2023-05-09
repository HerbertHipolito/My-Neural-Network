# Artificial Neural Network (ANN)
A simple implementation of a Multi-Layer Perceptron (MLP). Additional features were also implemented such as momentum and Early Stopping.
# GridSearch
In this project, I still used some libraries to help me as Numpy, Matplotlib, and Sklearn for the GridSearch, metrics (accuracy, precision, recall, and F1), and comparison of results. To measure my model performance, I carry out a small set of simulations with different hyperparameters to achieve the best results possible. The same tests were also performed in Sklearn ANN in order to compare both models and figure out how good my model was. The hyperparameters utilized are described in "parameters.py" => hyperparameter dict variable.
#  Results
Unfortunately, the comparisons are not totally fair because I did not implement all parameters used in the Sklearn ANN and the ANN contains a lot of internal randomness, for example, the way that the initial weights are initialized and data scrambling during at each epoch. However, my ANN had only 1% of metric results below the Sklearn ANN on average. The following graphs help us to view it.

## 20 neurons, 2 hidden leayer, momentum  0.4, learning Rate = 0.001

![plot_4](https://user-images.githubusercontent.com/94997683/236550267-551b36e9-defb-4f51-8190-02817ca1e13f.png)

## 25 neurons, 2 hidden leayer, momentum  0, learning Rate = 0.001

![plot_2](https://user-images.githubusercontent.com/94997683/236551117-150b9f81-34ac-4624-9fdf-b0b696b7495b.png)

# Results in test dataset

The dataset was generated from sklearn.datasets.make_classification to train and test the models. 
It selected 5 features (x) and 2 classes (y) with 10000 samples. The dataset was split into trainig and test(30%) datasets.
It carried out 24 simulations (GridSearch) according to the hyperparameters. The table below displays the results of all simulations.

| simulation | accuracy | precision | recall | f1     |
|------------|----------|-----------|--------|--------|
| 0          | 0.9540   | 0.9355    | 0.9707 | 0.9528 |
| 1          | 0.9527   | 0.9294    | 0.9739 | 0.9512 |
| 2          | 0.9537   | 0.9341    | 0.9713 | 0.9524 |
| 3          | 0.9537   | 0.9422    | 0.9636 | 0.9528 |
| 4          | 0.9520   | 0.9429    | 0.9596 | 0.9512 |
| 5          | 0.9547   | 0.9362    | 0.9714 | 0.9535 |
| 6          | 0.9537   | 0.9375    | 0.9681 | 0.9525 |
| 7          | 0.9537   | 0.9301    | 0.9753 | 0.9522 |
| 8          | 0.9533   | 0.9368    | 0.9681 | 0.9522 |
| 9          | 0.9533   | 0.9415    | 0.9635 | 0.9524 |
| 10         | 0.9543   | 0.9362    | 0.9707 | 0.9531 |
| 11         | 0.9523   | 0.9362    | 0.9667 | 0.9512 |
| 12         | 0.9520   | 0.9335    | 0.9686 | 0.9507 |
| 13         | 0.9530   | 0.9402    | 0.9642 | 0.9520 |
| 14         | 0.9527   | 0.9335    | 0.9700 | 0.9514 |
| 15         | 0.9520   | 0.9160    | 0.9863 | 0.9498 |
| 16         | 0.9517   | 0.9301    | 0.9712 | 0.9502 |
| 17         | 0.9513   | 0.9241    | 0.9766 | 0.9496 |
| 18         | 0.9510   | 0.9422    | 0.9583 | 0.9502 |
| 19         | 0.9530   | 0.9402    | 0.9642 | 0.9520 |
| 20         | 0.9537   | 0.9355    | 0.9700 | 0.9524 |
| 21         | 0.9520   | 0.9335    | 0.9686 | 0.9507 |
| 22         | 0.9530   | 0.9308    | 0.9733 | 0.9516 |
| 23         | 0.9533   | 0.9315    | 0.9733 | 0.9519 | 

The Average difference between my MLP and Sklearn MLP in relation to the 4 metrics.

|         | acc        | precision  | recall     | f1         |
| ------- | ----------| ----------| ----------| ----------|
| Average | -0.000676 | 0.005443  | -0.006753 | -0.000399 |
