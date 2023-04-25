from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import math
import os


class myNeuralNetwork:

  def __init__(self,learningRate,epoch,neuronNumber,weightsInitialValue,activeFunction,lostFunction,layerNumber=1,showProgress=False,momentum=0):

    self.learningRate = learningRate
    self.epoch = epoch
    self.neuronNumber = neuronNumber
    self.weightsInitialValue = weightsInitialValue
    self.layerNumber = layerNumber
    self.activeFunction = activeFunction
    self.lostFunction = lostFunction
    self.layers = []
    self.showProgress = showProgress
    self.momentum = momentum
  
  def initializeWeights(self):

    self.matrix = np.zeros((len(self.x[0]),self.neuronNumber))
    self.weights = []
    
    for i in range(len(self.x[0])):
      for j in range(self.neuronNumber):
        self.matrix[i,j] = self.weightsInitialValue()
    
    self.weights.append(self.matrix)
    
    for _ in range(self.layerNumber-1): #setting up the weights for the hidden layers
        self.matrix = np.zeros((self.neuronNumber,self.neuronNumber))
        for i in range(self.neuronNumber):
            for j in range(self.neuronNumber):
                self.matrix[i,j] = self.weightsInitialValue()
                
        self.weights.append(self.matrix)
                
    self.matrix = np.zeros((self.neuronNumber,2))

    for i in range(self.neuronNumber): # binary classification
      for j in range(2):
        self.matrix[i,j] = self.weightsInitialValue()

    self.weights.append(self.matrix)
  
  def trainingData(self,x,y):

    self.x = x
    self.y = y
    self.initializeWeights()

  def fit(self,validation_size=0.3):

    history = {
          'validation_acc':[],
          'training_acc':[]
      }

    #Setting up a matrix for all delta weights from the previous iteration to calculate the momentum.
    self.delta_weights_previous_iteration = []
    self.delta_weights_previous_iteration.append(np.zeros((len(self.x[0]),self.neuronNumber)))
    for _ in range(self.layerNumber-1): self.delta_weights_previous_iteration.append(np.zeros((self.neuronNumber,self.neuronNumber)))  
    self.delta_weights_previous_iteration.append(np.zeros((self.neuronNumber,2)))

    for i in range(self.epoch):

      prediction_training, prediction_validation = [],[]
      x_train, x_validation, y_train, y_validation = train_test_split(self.x,self.y,test_size=validation_size)
      self.y_train = y_train
      
      if self.showProgress:
        #os.system('cls') 
        print(f'----- In epoch {i+1} -----')

      for j in range(len(x_train)):

        self.layers ,self.finalResult ,result = [],[],[]

        self.layers.append(x_train[j])

        result = np.dot(x_train[j],self.weights[0])
        result = [self.activeFunction(element) for element in result]
        self.layers.append(result)
        
        for layer_index in range(self.layerNumber-1):#check here
            result = np.dot(result,self.weights[layer_index+1])
            result = [self.activeFunction(element) for element in result]
            self.layers.append(result)

        result = np.dot(result,self.weights[self.layerNumber])
        self.finalResult = [self.activeFunction(element) for element in result]

        self.layers.append(self.finalResult)

        prediction_training.append(1) if self.finalResult[1]>self.finalResult[0] else prediction_training.append(0)

        self.current_iteration = j
        self.update_weight()

      prediction_validation = self.predict(x_validation)

      history['validation_acc'].append(accuracy_score(prediction_validation,y_validation))
      history['training_acc'].append(accuracy_score(prediction_training,y_train))


    return prediction_training,history
    

  def predict(self,to_predict):

    results = []

    for row in to_predict:

      layer = [self.activeFunction(element) for element in np.dot(row,self.weights[0])]
      
      for index,weight in enumerate(self.weights):
        if index>0: layer = [self.activeFunction(element) for element in np.dot(layer,weight)]

      results.append(1) if layer[1]>layer[0] else results.append(0)
    
    return results
    

  def update_weight(self):

    #Setting up a matrix for all deltas
    delta_matrix = []
    delta_matrix.append(np.zeros((len(self.x[0]),self.neuronNumber)))
    for _ in range(self.layerNumber-1): delta_matrix.append(np.zeros((self.neuronNumber,self.neuronNumber)))  
    delta_matrix.append(np.zeros((self.neuronNumber,2)))
    real_value = [0,1] if self.y_train[self.current_iteration] == 1 else [1,0]


    for index in range(len(self.weights)-1,-1,-1):
      
      for i,row_weight in enumerate(self.weights[index]):

        for j in range(len(self.weights[index][i])):

          if index == (len(self.weights)-1): #Updating the last layer weights.
            delta = (self.layers[index+1][j]-real_value[j])*self.layers[index+1][j]*(1+self.layers[index+1][j])
            
          else: #Updating the layer weights left.
            sum_delta = 0
            for v,element in enumerate(delta_matrix[index+1][j]):
              sum_delta+=element*self.weights[index+1][j,v]
            
            delta = self.layers[index+1][j]*(1+self.layers[index+1][j])*sum_delta
            
          delta_matrix[index][i,j] = delta
          self.weights[index][i,j] -= self.learningRate*delta*self.layers[index][i] + self.momentum*self.delta_weights_previous_iteration[index][i,j]
          self.delta_weights_previous_iteration[index][i,j] = self.learningRate*delta*self.layers[index][i]
          
#https://optimization.cbe.cornell.edu/index.php?title=Momentum
