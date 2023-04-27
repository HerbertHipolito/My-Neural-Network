import math
import numpy as np

parameters = {
    'epoch':200,
    'learningRate':0.001,
    'neuronNumber':25,
    'activeFunction':'leaky_relu',
    'weightsInitialValue':lambda x=None:np.random.normal(loc=0,scale=0.05),
    'lostFunction':lambda y,y_hat: (y-y_hat)**2,
    'layerNumber':2,
    'showProgress':True,
    'momentum':0
}

graph_options = {
    'validation_x_traning':True,
    'hist':False
}

early_stopping_config = {
    'no_improvement_max':30,
    'e':0.001
}

