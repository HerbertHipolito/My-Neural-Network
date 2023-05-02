import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.neural_network import MLPClassifier

import numpy as np
from parameters import parameters,graph_options,early_stopping_config
from NeuralNetwork import myNeuralNetwork as MNN
import time

x,y = make_classification(n_samples=7500,n_classes=2,n_features=5,random_state=36)

#x = [[np.random.uniform(0,50),np.random.uniform(0,50)] for tupl in range(5000)]
#y = [1 if element[0]>=20 and element[1]>=12 else 0 for element in x]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

print('-------initializing the sklearn MLP-------')

start = time.time()

sklearn_model = MLPClassifier(solver='sgd',nesterovs_momentum=False,hidden_layer_sizes=[25,25],power_t=0,momentum=parameters['momentum'],beta_1=0,beta_2=0,batch_size=1,early_stopping=True,validation_fraction=0.3,alpha=0,verbose=True).fit(x_train,y_train)
y_predicted_sklearn = sklearn_model.predict(x_test) 

print('Acc:'+str(round(accuracy_score(y_predicted_sklearn,y_test),3)))
print('Recall: '+str(round(recall_score(y_predicted_sklearn,y_test),3)))
print('Precision: '+str(round(precision_score(y_predicted_sklearn,y_test),3)))
print('f1: '+str(round(f1_score(y_predicted_sklearn,y_test),3)))

sklearn_time = time.time() - start
print(f'Sklearn mlp used {sklearn_time} of time')

print('-------Initializing my MLP-------')

start = time.time()

model = MNN(**parameters)
model.trainingData(x_train,y_train)
model.set_early_stopping(early_stopping_config)
print(parameters)
training_prediction,history = model.fit()  

my_mlp_time = time.time() - start
print(f'My mlp used {my_mlp_time} of time')
print('\n-------Performing the predictions-------\n')

y_predicted = model.predict(x_test)
print('Acc:'+str(round(accuracy_score(y_predicted,y_test),3)))
print('Recall: '+str(round(recall_score(y_predicted,y_test),3)))
print('Precision: '+str(round(precision_score(y_predicted,y_test),3)))
print('f1: '+str(round(f1_score(y_predicted,y_test),3)))

print('\n-------Comparing the models -------\n')

print('Acc:'+str(round(accuracy_score(y_predicted,y_test),3)-round(accuracy_score(y_predicted_sklearn,y_test),3) ))
print('Recall: '+str(round(recall_score(y_predicted,y_test),3)-round(recall_score(y_predicted_sklearn,y_test),3) ))
print('Precision: '+str(round(precision_score(y_predicted,y_test),3)-round(precision_score(y_predicted_sklearn,y_test),3)))
print('f1: '+str(round(f1_score(y_predicted,y_test),3)-round(f1_score(y_predicted_sklearn,y_test),3)))

print('Difference of time between the models: '+str(my_mlp_time - sklearn_time))

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
    q=500
    for i in range(5001,30000-q,q):

        #x2 = [[np.random.uniform(0,50),np.random.uniform(0,50)] for _ in range(1000)]
        #y2 = [1 if element[0]>=20 and element[1]>=12 else 0 for element in x2]
        #x2,y2 = make_classification(n_samples=500,n_classes=2,n_features=5,random_state=27)
        prediction = model.predict(x_total[i:i+q,:])
        final_result.append(accuracy_score(y_total[i:i+q],prediction))

    plt.hist(final_result)
    plt.show()
    print(np.array(final_result).mean())
    print(np.array(final_result).std())


