from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np
import math
import os

import pandas as pd
import numpy as np
import math
import os


class myNeuralNetwork:

  def __init__(self,learningRate,epoch,neuronNumber,weightsInitialValue,activeFunction,lostFunction,layerNumber=1,showProgress=False,momentum=0,regularization_l1=0,regularization_l2=0,show_validation_traning_acc=False):

    self.learningRate = learningRate
    self.epoch = epoch
    self.neuronNumber = neuronNumber
    self.weightsInitialValue = weightsInitialValue
    self.layerNumber = layerNumber
    self.lostFunction = lostFunction
    self.layers = []
    self.showProgress = showProgress
    self.momentum = momentum
    self.use_early_stopping = False
    self.activeFunctionName = activeFunction
    self.regularization_l1 = regularization_l1
    self.regularization_l2 = regularization_l2
    self.show_validation_traning_acc = show_validation_traning_acc
    self.bias = []

    if activeFunction == "sigmoid":
      self.activeFunction = lambda x:1/(1+(math.e**(-x)))
      self.derivateOfActiveFunction = lambda x:x*(1+x)
    elif activeFunction == "relu":
      self.activeFunction = lambda x:max(0,x)
      self.derivateOfActiveFunction = lambda x:1 if x>=0 else 0
    elif activeFunction == "leaky_relu":
      self.activeFunction = lambda x:x if x>=0 else 0.01*x
      self.derivateOfActiveFunction = lambda x: 1 if x>=0 else 0.01
    elif activeFunction == "soft_plus":
      self.activeFunction = lambda x: math.log(1+math.e**x)
      self.derivateOfActiveFunction = lambda x: 1/(1+math.e**(x))
    else:
      raise Exception('Activation function not found')
      
  def initializeWeightsBias(self):

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
                
    self.matrix = np.zeros((self.neuronNumber,2)) # binary classification
    for i in range(self.neuronNumber): 
      for j in range(2):
        self.matrix[i,j] = self.weightsInitialValue()

    self.weights.append(self.matrix)

    for i in range(self.layerNumber): self.bias.append(np.zeros(self.neuronNumber)) # initializing bias
    self.bias.append(np.zeros(2))
    
 
  def trainingData(self,x,y):

    self.x = x
    self.y = y
    self.columnNumber = len(x[0])
    self.initializeWeightsBias()

  def fit(self,validation_size=0.1):

    history = {
          'validation_acc':[],
          'training_acc':[]
      }
    best_acc = 0
    best_weights = self.weights

    self.delta_weights_previous_iteration = []  #Setting up a matrix for all delta weights from the previous iteration to calculate the momentum
    self.delta_weights_previous_iteration.append(np.zeros((len(self.x[0]),self.neuronNumber)))
    for _ in range(self.layerNumber-1): self.delta_weights_previous_iteration.append(np.zeros((self.neuronNumber,self.neuronNumber)))  
    self.delta_weights_previous_iteration.append(np.zeros((self.neuronNumber,2)))

    for i in range(self.epoch):

      x_train, x_validation, y_train, y_validation = train_test_split(self.x,self.y,test_size=validation_size)
      prediction_training, prediction_validation = np.zeros(len(x_train)),[]
      self.y_train = y_train
      
      if self.showProgress:
        #os.system('cls') 
        print(f'----- In epoch {i+1} -----')

      for j in range(len(x_train)):

        self.layers, self.finalResult, result = [],[],[]

        self.layers.append(x_train[j])

        result = [self.activeFunction(element+self.bias[0][i]) for i,element in enumerate(np.dot(x_train[j],self.weights[0]))]
        self.layers.append(result)
        
        for layer_index in range(self.layerNumber-1):
            result = [self.activeFunction(element+self.bias[layer_index+1][i]) for i,element in enumerate(np.dot(result,self.weights[layer_index+1]))]
            self.layers.append(result)

        self.finalResult = [self.activeFunction(element+self.bias[self.layerNumber][i]) for i,element in enumerate(np.dot(result,self.weights[self.layerNumber]))]

        self.layers.append(self.finalResult)

        #prediction_training.append(1) if self.finalResult[1]>self.finalResult[0] else prediction_training.append(0)
        if self.finalResult[1]>self.finalResult[0]: prediction_training[j] = 1

        self.current_iteration = j
        self.update_weight()

      prediction_validation = self.predict(x_validation)

      current_validation_acc = accuracy_score(prediction_validation,y_validation)
      current_training_acc = accuracy_score(prediction_training,y_train)

      history['validation_acc'].append(current_validation_acc)
      history['training_acc'].append(current_training_acc)

      if self.show_validation_traning_acc:
        print(current_validation_acc)
        print(current_training_acc)

      if self.use_early_stopping:
        if current_validation_acc - self.e >= best_acc:
          count_early_stopping = 0
          best_acc = current_validation_acc
          best_weights = self.weights
        else:
          count_early_stopping +=1
          if self.show_no_improviment_message:  print(f"No improviment found {count_early_stopping}")
          if count_early_stopping >=self.no_improvement_times:
            if not self.return_lastest_weights: self.weights = best_weights
            return prediction_training,history
          
    return prediction_training,history
 
  def predict(self,to_predict):

    results = np.zeros(len(to_predict))

    for index,row in enumerate(to_predict):

      layer = [self.activeFunction(element + self.bias[0][index]) for index,element in enumerate(np.dot(row,self.weights[0]))]
      
      for i,weight in enumerate(self.weights):
        if i>0: layer = [self.activeFunction(element + self.bias[i][j]) for j,element in enumerate(np.dot(layer,weight))]

      #results.append(1) if layer[1]>layer[0] else results.append(0) # optimize this part of code (remove the dinamyc allocation).
      if layer[1]>layer[0]: results[index] = 1
    
    return results
  
  def set_early_stopping(self,early_stopping_config):

    self.use_early_stopping = True
    self.no_improvement_times = early_stopping_config['no_improvement_max']
    self.e = early_stopping_config['e']
    self.return_lastest_weights = early_stopping_config['return_lastest_weights']
    self.show_no_improviment_message =  early_stopping_config['show_no_improviment_message'] 

    print(f"Early stopping activated with x = {early_stopping_config['no_improvement_max']} and e = {early_stopping_config['e']}")

  def update_weight(self):

    #Setting up a matrix for all deltas
    delta_matrix = []
    delta_matrix.append(np.zeros((self.columnNumber,self.neuronNumber)))
    for _ in range(self.layerNumber-1): delta_matrix.append(np.zeros((self.neuronNumber,self.neuronNumber)))  
    delta_matrix.append(np.zeros((self.neuronNumber,2)))
    real_value = [0,1] if self.y_train[self.current_iteration] == 1 else [1,0]
    
    weight_layer = len(self.weights) - 1

    for index in range(weight_layer,-1,-1):
      
      for i,row_weight in enumerate(self.weights[index]):

        len_weights = len(self.weights[index][i])

        for j in range(len_weights): 

          if index == (weight_layer): #Updating the last layer weights.
            l1 = self.regularization_l1*np.sign(self.weights[index][i,j])
            l2 = 2*self.regularization_l2*self.weights[index][i,j]
            delta = (self.layers[index+1][j]-real_value[j])*self.derivateOfActiveFunction(self.layers[index+1][j]) + l1 + l2 # l1 and l2 regularization added
            
          else: #Updating the layer weights left.
            sum_delta = 0
            for v,element in enumerate(delta_matrix[index+1][j]):
              sum_delta+=element*self.weights[index+1][j,v]
            
            delta = self.derivateOfActiveFunction(self.layers[index+1][j])*sum_delta
            
          delta_matrix[index][i,j] = delta
          self.weights[index][i,j] -= self.learningRate*delta*self.layers[index][i] + self.momentum*self.delta_weights_previous_iteration[index][i,j] # momentum added
          self.bias[index][j] -= self.learningRate*delta #updating Bias
          
          self.delta_weights_previous_iteration[index][i,j] = self.learningRate*delta*self.layers[index][i] 

