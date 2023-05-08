import math
import numpy as np

parameters = { # parameters for only one test
    'epoch':200,
    'learningRate':0.001,
    'neuronNumber':25,
    'activeFunction':'relu',
    'weightsInitialValue':lambda x=None:np.random.normal(loc=0,scale=0.05),
    'lostFunction':lambda y,y_hat: (y-y_hat)**2, # this parameter has not been used. it just let you know what lost function was implemented.
    'layerNumber':2,
    'showProgress':True,
    'momentum':0,
    'regularization_l1':0,
    'regularization_l2':0,
    'show_validation_traning_acc':True
}

hyperparameters = {
    'epoch':[200],
    'learningRate':[0.001],
    'neuronNumber':[15,20,25,30],
    'weightsInitialValue':[lambda x=None:np.random.normal(loc=0,scale=0.05)],
    'activeFunction':['relu'],
    'lostFunction':[lambda y,y_hat: (y-y_hat)**2], # this parameter has not been used. it just let you know what lost function was implemented.
    'layerNumber':[2,3],
    'showProgress':[True],
    'regularization_l1':[0],
    'regularization_l2':[0],
    'momentum':[0,0.4,0.8],
    'show_validation_traning_acc':[False]
}

graph_options = {
    'validation_x_traning':True,
    'hist':False
}

early_stopping_config = {
    'e':0.001,
    'no_improvement_max':10,
    'return_lastest_weights':True,    # False means that the weights of the best accucacy model will be setting.
    'show_no_improviment_message':False
} 
