from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np
import math
import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import math
import os

# return the weights that obtained the best acc

class myNeuralNetwork:

  def __init__(self,learningRate,epoch,neuronNumber,weightsInitialValue,activeFunction,lostFunction,layerNumber=1,showProgress=False,momentum=0,regularization_l1=0,regularization_l2=0):

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
    best_acc = 0
    best_weights = self.weights

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
        
        for layer_index in range(self.layerNumber-1):
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

      current_validation_acc = accuracy_score(prediction_validation,y_validation)
      current_training_acc = accuracy_score(prediction_training,y_train)

      history['validation_acc'].append(current_validation_acc)
      history['training_acc'].append(current_training_acc)

      if self.use_early_stopping:
        if current_validation_acc - self.e >= best_acc:
          count_early_stopping = 0
          best_acc = current_validation_acc
          best_weights = self.weights
        else:
          count_early_stopping +=1
          print(f"No improviment found {count_early_stopping}")
          if count_early_stopping >=self.no_improvement_times:
            if not self.return_lastest_weights: self.weights = best_weights
            return prediction_training,history
          
    return prediction_training,history
    

  def predict(self,to_predict):

    results = []

    for row in to_predict:

      layer = [self.activeFunction(element) for element in np.dot(row,self.weights[0])]
      
      for index,weight in enumerate(self.weights):
        if index>0: layer = [self.activeFunction(element) for element in np.dot(layer,weight)]

      results.append(1) if layer[1]>layer[0] else results.append(0)
    
    return results
  
  def set_early_stopping(self,early_stopping_config):

    self.use_early_stopping = True
    self.no_improvement_times = early_stopping_config['no_improvement_max']
    self.e = early_stopping_config['e']
    self.return_lastest_weights = early_stopping_config['return_lastest_weights']

    print(f"Early stopping activated with x = {early_stopping_config['no_improvement_max']} and e = {early_stopping_config['e']}")

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
          self.delta_weights_previous_iteration[index][i,j] = self.learningRate*delta*self.layers[index][i]

    
          
class  normalization():
  
  def __init__(self,dataset):
    
    if not isinstance(dataset, np.ndarray): raise Exception("Dataset must be numpy type")
        
    self.dataset = dataset
    self.max_array = []
    self.min_array = []
  
  def fit(self):
    
    normalized_dataset = np.zeros(self.dataset.shape)
    
    for j in range(self.dataset.shape[1]):
      
      current_column = self.dataset[:,j]
      column_min =  current_column.min()
      column_max =  current_column.max()
      
      self.max_array.append(self.max_array)
      self.min_array.append(self.min_array)
      
      for i,element in enumerate(current_column):
        
        normalized_dataset[i,j] = (element - column_min)/(column_max - column_min)
    
    return normalized_dataset
  
  def transform(self,dataset2):
    
    normalized_dataset2 = np.zeros(dataset2.shape)
    
    for j in range(dataset2.shape[1]):
      
      current_column = dataset2[:,j]
      
      for i,element in enumerate(current_column):
        
        normalized_dataset2[i,j] = (element - self.min_array[j])/(self.max_array[j]-self.min_array[j])
        
    return normalized_dataset2