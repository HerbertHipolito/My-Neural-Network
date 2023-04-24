import math
import numpy as np

parameters = {
    'epoch':350,
    'learningRate':0.001,
    'neuronNumber':20,
    'activeFunction':lambda x:1/(1+(math.e**(-x))),
    'weightsInitialValue':lambda x=None:np.random.normal(loc=0,scale=0.05),
    'lostFunction':lambda y,y_hat: (y-y_hat)**2,
    'layerNumber':2,
    'showProgress':True
}

graph_options = {
    'validation_x_traning':True,
    'hist':True
}