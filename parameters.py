import math
import numpy as np

parameters = {
    'epoch':200,
    'learningRate':0.001,
    'neuronNumber':25,
    'activeFunction':'leaky_relu',
    'weightsInitialValue':lambda x=None:np.random.normal(loc=0,scale=0.05),
    'lostFunction':lambda y,y_hat: (y-y_hat)**2, # this parameter has not been used. it just let you know what lost function was implemented.
    'layerNumber':2,
    'showProgress':True,
    'momentum':0.2,
    'regularization_l1':0,
    'regularization_l2':0.4
}

graph_options = {
    'validation_x_traning':True,
    'hist':True
}

early_stopping_config = {
    'e':0.001,
    'no_improvement_max':20,
    'return_lastest_weights':True  # False means that the weights of the best accucacy model wil be setting.
} 