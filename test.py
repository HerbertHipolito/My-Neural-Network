import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score

import math
import numpy as np
from parameters import parameters,graph_options
from NeuralNetwork import myNeuralNetwork as MNN


#x,y = make_classification(n_samples=3000,n_classes=2,n_features=5,random_state=27)
x = [[np.random.uniform(0,50),np.random.uniform(0,50)] for tupl in range(3000)]
y = [1 if element[0]>=20 and element[1]>=12 else 0 for element in x]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

print('-------Initializing the model-------')

model = MNN(**parameters)
model.trainingData(x_train,y_train)
training_prediction,history = model.fit()  

print('\n-------Performing the predictions-------\n')

y_predicted = model.predict(x_test)
print('Acc:'+str(round(accuracy_score(y_predicted,y_test),3)))
print('Recall:'+str(round(recall_score(y_predicted,y_test),3)))
print('Precision'+str(round(precision_score(y_predicted,y_test),3)))

print('\n-------plotting the training and validation acc-------\n')

plt.figure(figsize=(10,6))
plt.plot(history['training_acc'],label='training')
plt.plot(history['validation_acc'],label='validation')
plt.grid()
plt.legend()
plt.title('training x validation')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

if graph_options['hist']:
    
    print('\n-------Plotting a histogram of a set of simulations-------\n')

    final_result = []

    for test in range(200):

        x2 = [[np.random.uniform(0,50),np.random.uniform(0,50)] for _ in range(1000)]
        y2 = [1 if element[0]>=20 and element[1]>=12 else 0 for element in x2]
        #x2,y2 = make_classification(n_samples=500,n_classes=2,n_features=5,random_state=30)
        prediction = model.predict(x2)
        final_result.append(accuracy_score(y2,prediction))

    plt.hist(final_result)
    plt.show()
    print(np.array(final_result).mean())
    print(np.array(final_result).std())

