from NeuralNetwork import myNeuralNetwork as MNN
from parameters import hyperparameters,early_stopping_config

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

import json
import matplotlib.pyplot as plt
import time

x,y = make_classification(n_samples=10000,n_classes=2,n_features=5,random_state=36)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
results = dict()

for index,simulation in enumerate(list(ParameterGrid(hyperparameters))):
    
    print(f'-------Initializing {index+1} simulation-------')
    print(simulation)
    print('-------initializing the sklearn MLP-------')

    start = time.time()

    sklearn_model = MLPClassifier(solver='sgd',nesterovs_momentum=False,hidden_layer_sizes=[simulation['neuronNumber'] for _ in range(simulation['layerNumber'])],power_t=0,momentum=simulation['momentum'],beta_1=0,beta_2=0,batch_size=1,early_stopping=True,validation_fraction=0.1,alpha=simulation['regularization_l2'],verbose=True).fit(x_train,y_train)
    y_predicted_sklearn = sklearn_model.predict(x_test) 

    sklearn_time = time.time() - start

    print(sklearn_model.validation_scores_)
    print(f'Sklearn mlp used {sklearn_time} of time')

    acc_sklearn = accuracy_score(y_test,y_predicted_sklearn)
    recall_sklearn = recall_score(y_test,y_predicted_sklearn)
    precision_sklearn = precision_score(y_test,y_predicted_sklearn)
    f1_sklearn = f1_score(y_test,y_predicted_sklearn)

    print('Acc:'+str(round(acc_sklearn ,4)))
    print('Recall: '+str(round(recall_sklearn,4)))
    print('Precision: '+str(round(precision_sklearn,4)))
    print('f1: '+str(round(f1_sklearn,4)))

    print('-------initializing my MLP-------')

    start = time.time()

    model = MNN(**simulation)
    model.trainingData(x_train,y_train)
    model.set_early_stopping(early_stopping_config)

    training_prediction,history = model.fit()  
    y_predicted = model.predict(x_test)

    my_mlp_time = time.time() - start

    acc = round(accuracy_score(y_test,y_predicted),4)
    recall = round(recall_score(y_test,y_predicted),4)
    precision = round(precision_score(y_test,y_predicted),4)
    f1 = round(f1_score(y_test,y_predicted),4)

    print('Acc:'+str(acc))
    print('Recall: '+str(recall))
    print('Precision: '+str(precision))
    print('f1: '+str(f1))
    
    results[str(index)] = {
        'accuracy':acc,
        'precision':precision,
        'recall':recall,
        'f1':f1,
        'acc_sklearn':acc_sklearn,
        'recall_sklearn':recall_sklearn,
        'precision_sklearn':precision_sklearn,
        'f1_sklearn':f1_sklearn,
        'hyperparameters':{
            'momentum':simulation['momentum'],
            'l1':simulation['regularization_l1'],
            'l2':simulation['regularization_l2'],
            'neuronNumber':simulation['neuronNumber'],
            'layerNumber':simulation['layerNumber']
        },
        'training_acc':history['training_acc'],
        'validation_acc':history['validation_acc'],
        'sklearn_validation_acc':sklearn_model.validation_scores_,
        'sklearn_time':sklearn_time,
        'my_mlp_time':my_mlp_time,
        'time_difference':sklearn_time-my_mlp_time,
        'index':index
    }

    plt.title('Sim_'+str(index)+' momen='+str(simulation['momentum'])+' act. func.: '+str(simulation['activeFunction'])+' N.Number='+str(simulation['neuronNumber'])+' N.Layer='+str(simulation['layerNumber'])+' L.rate='+str(simulation['learningRate']))
    plt.plot(history['validation_acc'],label='validation')
    plt.plot(history['training_acc'],label='training')
    plt.plot(sklearn_model.validation_scores_,label='sklearn Validation')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    plt.savefig(f'./all_plots/plot_'+str(index))
    plt.clf()

with open("results.json", "w") as outfile:
    json.dump(results, outfile)

