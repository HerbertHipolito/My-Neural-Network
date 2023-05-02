from NeuralNetwork import myNeuralNetwork as MNN
from parameters import hyperparameters,early_stopping_config
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.datasets import make_classification
import json
import matplotlib.pyplot as plt

x,y = make_classification(n_samples=3000,n_classes=2,n_features=5,random_state=36)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
results = dict()

for index,simulation in enumerate(list(ParameterGrid(hyperparameters))):
    
    print(f'-------Initializing {index+1} the model-------')
    print(simulation)
 
    model = MNN(**simulation)
    model.trainingData(x_train,y_train)
    model.set_early_stopping(early_stopping_config)

    training_prediction,history = model.fit()  
    y_predicted = model.predict(x_test)
    
    acc = round(accuracy_score(y_predicted,y_test),3)
    recall = round(recall_score(y_predicted,y_test),3)
    precision = round(precision_score(y_predicted,y_test),3)
    f1 = round(f1_score(y_predicted,y_test),3)

    print('Acc:'+str(acc))
    print('Recall: '+str(recall))
    print('Precision: '+str(precision))
    print('f1: '+str(f1))
    
    results[str(index)] = {
        'accuracy':acc,
        'precision':precision,
        'recall':recall,
        'f1':f1,
        'hyperparameters':{
            'momentum':simulation['momentum'],
            'l1':simulation['regularization_l1'],
            'l2':simulation['regularization_l2'],
            'neuronNumber':simulation['neuronNumber'],
            'layerNumber':simulation['layerNumber']
        },
        'index':index
    }

    plt.title('Simu_'+str(index)+' momen='+str(simulation['momentum'])+' L1='+str(simulation['regularization_l1'])+' L2='+str(simulation['regularization_l2'])+' N.Number='+str(simulation['neuronNumber'])+'N.Layer='+str(simulation['layerNumber'])+'L.rate'+str(simulation['learningRate']))
    plt.plot(history['validation_acc'],label='validation')
    plt.plot(history['training_acc'],label='training')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    plt.savefig(f'./all_plots/plot_'+str(index))
    plt.clf()

with open("results.json", "w") as outfile:
    json.dump(results, outfile)

