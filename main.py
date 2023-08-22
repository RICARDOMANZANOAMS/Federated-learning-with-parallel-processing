import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import mlflow
import os
import matplotlib.pyplot as plt
import time
import pandas as pd


#Define the neural network to train in parallelt
DIM_IN = 784    #Define input dimension
DIM_OUT = 10    #Define output dimension 

# Define your model class
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.layer1 = torch.nn.Linear(DIM_IN, 100)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(100, 100 )
        self.relu2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(100, 100 )
        self.relu3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Linear(100, DIM_OUT )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.layer4(x)
        
        return x

#Define the pre-processing of the dataset
def preprocess_dataset(fold):
    columns_to_select_no_label=fold.columns[fold.columns!='label']  #get the columns except label    
    features = fold.loc[:, columns_to_select_no_label] #select the columns except label
    labels=fold['label']   #select the label
    
    features_np=features.to_numpy() #transform to numpy   
    features_t=torch.from_numpy(features_np).float()  #transform to torch
    
    labels_np=labels.to_numpy()   #transform to numpy labels
    labels_t=torch.tensor(labels_np)  # transform to torch label
    return features_t,labels_t    #return features and labels


# Number of models to train
num_models = 5

# Create a list of models
seed=20   
torch.manual_seed(seed)    #specify the seed to start weigths in the neural network
#Create 6 models 
start_model_general=Classifier()  #model to overwrite the other models
model1=Classifier() 
model2=Classifier() 
model3=Classifier()
model4=Classifier()
model5=Classifier()
start_model_save_input=Classifier()   #model to save the general model


# Define the worker function for training
#This function will run in parallel

def train_worker(args):
    results_f1_score=[]    #array to save results
    both_models=[]         #array will contain the model receive before training, the model after training, and the results
                           #[Model_before_training, Model_after_training, Results]
    
    #We receive the following info in each node running in parallel
    #[model_to_train, data_features_to_train, labels_to_train, data_test_features, labels_to_test,number_of node]
    #model: this is the model received before trainin
    #data: this contains the features to train
    #target: this contains the labels that belongs to the data train
    #data_test: this contains the dataset for testing. It only contains the test 
    #target_test: this contains the labels that belong to the features
    #node: it is the number of node that it is processing the data

    model, data, target,data_test,target_test,node= args    

    start_model_save_input.load_state_dict(model.state_dict())  #Assigned the weights and biases to start_model_save_input. This var start_model_save_input will be sent as output of this function
    criterion= nn.CrossEntropyLoss()      #Define the loss
    optimizer= torch.optim.Adam(model.parameters(), lr=1e-4)  #Define the optimizer

    start_time = time.time()     #Measure time of training 
    epochs=100                   #Number of epochs that will train the data inside each node before concatenation
    for epoch in range(epochs):  #Training process
        optimizer.zero_grad()    #Zero grad
        output = model(data)     #Pass the data through the model
        loss = criterion(output, target) #Find the loss. It is not necessary to put softmax because we are using crossentrophy
        loss.backward()        # Loss backward
        optimizer.step()      #Optimizer step

    end_time = time.time()    #measure training time
    execution_time = end_time - start_time    #Get execution time
  
    #Evaluation of the model after training
    with torch.no_grad():            #The model is not trained in evaluation mode
        outputs = model(data_test)  #predict the data_test output
        
        probabilities = nn.functional.softmax(outputs, dim=1)  #find prob because we are not using cross_entrophy
        _, predicted = torch.max(probabilities, 1)    #find the predictions
        
        pred_numpy=predicted.data.cpu().numpy()    #transform prediction to numpy from gpu
        labels_numpy=target_test.data.cpu().numpy()  #transform labels to numpy from gpu
        
       
    cf_matrix = confusion_matrix(pred_numpy, labels_numpy)  #find confusion matrix
    target_names=['0','1','2','3','4','5','6','7','8','9']  
    #print(cf_matrix)
    report=classification_report(pred_numpy, labels_numpy,target_names=target_names,zero_division=0,output_dict=True) #find classification report
    #print(report)
    
    #find f1-score for each class
    macro_f1 = report['macro avg']['f1-score']
    class_0_f1 = report['0']['f1-score']
    class_1_f1 = report['1']['f1-score']
    class_2_f1 = report['2']['f1-score']
    class_3_f1 = report['3']['f1-score']
    class_4_f1 = report['4']['f1-score']
    class_5_f1 = report['5']['f1-score']
    class_6_f1 = report['6']['f1-score']
    class_7_f1 = report['7']['f1-score']
    class_8_f1 = report['8']['f1-score']
    class_9_f1 = report['9']['f1-score']
   

    print("macro")
    print(macro_f1)
    # Keep track of results in each node    
    results_f1_score.append(node )
    results_f1_score.append(macro_f1)
    results_f1_score.append(class_0_f1)
    results_f1_score.append(class_1_f1)
    results_f1_score.append(class_2_f1)
    results_f1_score.append(class_3_f1)
    results_f1_score.append(class_4_f1)
    results_f1_score.append(class_5_f1)
    results_f1_score.append(class_6_f1)
    results_f1_score.append(class_7_f1)
    results_f1_score.append(class_8_f1)
    results_f1_score.append(class_9_f1)
    results_f1_score.append(execution_time)

    #Append to return only one output
    both_models.append(start_model_save_input)  #append model before training
    both_models.append(model)                   #append model after training
    both_models.append(results_f1_score)        #append results tested in test dataset
    return both_models

if __name__ == '__main__':
    
    results_epoch_f1_score=[]
    
    #Create datasets
    

    fold1_train=pd.read_csv('C:/RICARDO/2023 CISTECH BACKUP/cistech/2023/MNIST DATASET FEDERATED IN ONE COMPUTER/test federated learning mnist/Mnist dataset zero day attack/Dataset 5 folds no two classes in each fold/fold1_train_mnist_5_folds.csv')
    fold1_train=fold1_train.iloc[:,1:]  #eliminate the first column that contains the count for each row
    fold2_train=pd.read_csv('C:/RICARDO/2023 CISTECH BACKUP/cistech/2023/MNIST DATASET FEDERATED IN ONE COMPUTER/test federated learning mnist/Mnist dataset zero day attack/Dataset 5 folds no two classes in each fold/fold2_train_mnist_5_folds.csv')
    fold2_train=fold2_train.iloc[:,1:]
    fold3_train=pd.read_csv('C:/RICARDO/2023 CISTECH BACKUP/cistech/2023/MNIST DATASET FEDERATED IN ONE COMPUTER/test federated learning mnist/Mnist dataset zero day attack/Dataset 5 folds no two classes in each fold/fold3_train_mnist_5_folds.csv')
    fold3_train=fold3_train.iloc[:,1:]
    fold4_train=pd.read_csv('C:/RICARDO/2023 CISTECH BACKUP/cistech/2023/MNIST DATASET FEDERATED IN ONE COMPUTER/test federated learning mnist/Mnist dataset zero day attack/Dataset 5 folds no two classes in each fold/fold4_train_mnist_5_folds.csv')
    fold4_train=fold4_train.iloc[:,1:]
    fold5_train=pd.read_csv('C:/RICARDO/2023 CISTECH BACKUP/cistech/2023/MNIST DATASET FEDERATED IN ONE COMPUTER/test federated learning mnist/Mnist dataset zero day attack/Dataset 5 folds no two classes in each fold/fold5_train_mnist_5_folds.csv')
    fold5_train=fold5_train.iloc[:,1:]

    fold1_test=pd.read_csv('C:/RICARDO/2023 CISTECH BACKUP/cistech/2023/MNIST DATASET FEDERATED IN ONE COMPUTER/test federated learning mnist/Mnist dataset zero day attack/Dataset 5 folds no two classes in each fold/fold1_test_mnist_5_folds.csv')
    fold1_test=fold1_test.iloc[:,1:]
    fold2_test=pd.read_csv('C:/RICARDO/2023 CISTECH BACKUP/cistech/2023/MNIST DATASET FEDERATED IN ONE COMPUTER/test federated learning mnist/Mnist dataset zero day attack/Dataset 5 folds no two classes in each fold/fold2_test_mnist_5_folds.csv')
    fold2_test=fold2_test.iloc[:,1:]
    fold3_test=pd.read_csv('C:/RICARDO/2023 CISTECH BACKUP/cistech/2023/MNIST DATASET FEDERATED IN ONE COMPUTER/test federated learning mnist/Mnist dataset zero day attack/Dataset 5 folds no two classes in each fold/fold3_test_mnist_5_folds.csv')
    fold3_test=fold3_test.iloc[:,1:]
    fold4_test=pd.read_csv('C:/RICARDO/2023 CISTECH BACKUP/cistech/2023/MNIST DATASET FEDERATED IN ONE COMPUTER/test federated learning mnist/Mnist dataset zero day attack/Dataset 5 folds no two classes in each fold/fold4_test_mnist_5_folds.csv')
    fold4_test=fold4_test.iloc[:,1:]
    fold5_test=pd.read_csv('C:/RICARDO/2023 CISTECH BACKUP/cistech/2023/MNIST DATASET FEDERATED IN ONE COMPUTER/test federated learning mnist/Mnist dataset zero day attack/Dataset 5 folds no two classes in each fold/fold5_test_mnist_5_folds.csv')
    fold5_test=fold5_test.iloc[:,1:]

    print(fold1_train)
    #Pass the datasets through preprocess function
    #training dataset
    features_fold1_train,labels_fold1_train=preprocess_dataset(fold1_train) 
    features_fold2_train,labels_fold2_train=preprocess_dataset(fold2_train) 
    features_fold3_train,labels_fold3_train=preprocess_dataset(fold3_train) 
    features_fold4_train,labels_fold4_train=preprocess_dataset(fold4_train) 
    features_fold5_train,labels_fold5_train=preprocess_dataset(fold5_train) 

    #testing dataset
    features_fold1_test,labels_fold1_test=preprocess_dataset(fold1_test) 
    features_fold2_test,labels_fold2_test=preprocess_dataset(fold2_test) 
    features_fold3_test,labels_fold3_test=preprocess_dataset(fold3_test) 
    features_fold4_test,labels_fold4_test=preprocess_dataset(fold4_test) 
    features_fold5_test,labels_fold5_test=preprocess_dataset(fold5_test) 

    #create arrays to pass to the parallel function process
    folds_train_features=[features_fold1_train,features_fold2_train,features_fold3_train,features_fold4_train,features_fold5_train]
    folds_train_labels=[labels_fold1_train,labels_fold2_train,labels_fold3_train,labels_fold4_train,labels_fold5_train]

    folds_test_features=[features_fold1_test,features_fold2_test,features_fold3_test,features_fold4_test,features_fold5_test]
    folds_test_labels=[labels_fold1_test,labels_fold2_test,labels_fold3_test,labels_fold4_test,labels_fold5_test]

    #name nodes for output
    nodes=['node 1','node 2','node 3','node 4','node 5']
    number_rounds=400
    round_current=0  #
    
    #concat testing datasets 
    all_test_folds_to_evaluate_in_global=pd.concat([fold1_test,fold2_test,fold3_test,fold4_test,fold5_test])
    features_all_test_folds_to_evaluate_in_global,labels_all_test_folds_to_evaluate_in_global=preprocess_dataset(all_test_folds_to_evaluate_in_global) 

    # global model 
    global_model_eval_macro=[]  
    global_model_eval_class_0=[] 
    global_model_eval_class_1=[] 
    global_model_eval_class_2=[]
    global_model_eval_class_3=[] 
    global_model_eval_class_4=[] 
    global_model_eval_class_5=[] 
    global_model_eval_class_6=[]  
    global_model_eval_class_7=[]
    global_model_eval_class_8=[]
    global_model_eval_class_9=[]   

   
    while(round_current<number_rounds):

        results_f1_score=[] #array to save results
        
        #Initialize all the models with the same weights and biases
        model1.load_state_dict(start_model_general.state_dict())  #assigned the weights and biases in start_model_general to the model1
        model2.load_state_dict(start_model_general.state_dict())  #assigned the weights and biases in start_model_general to the model2
        model3.load_state_dict(start_model_general.state_dict())  #assigned the weights and biases in start_model_general to the model3
        model4.load_state_dict(start_model_general.state_dict())  #assigned the weights and biases in start_model_general to the model4
        model5.load_state_dict(start_model_general.state_dict())  #assigned the weights and biases in start_model_general to the model5
        models=[model1,model2,model3,model4,model5]  #put all the model in an array
              

        #We create the train_args. This part is the most important since we pass the arrays of models, datasets to each node 
        train_args = [(model, data, target,data_test,target_test,node) for  model, data, target,data_test,target_test,node in zip(models, folds_train_features, folds_train_labels,folds_test_features, folds_test_labels,nodes)]

        # Create a multiprocessing pool
        with multiprocessing.Pool(processes=num_models) as pool:
            # Train models concurrently and collect results
            all_models = pool.map(train_worker, train_args)
            pool.close()
            pool.join()
    

       
        #The next loops are used to extract the results in each node
        for idx, both_models_results in enumerate(all_models):
            
            for idy, model_result in enumerate(both_models_results):
                if idy==2:  #extract element in position 2
                    results_f1_score.append(model_result)
        
        results_epoch_f1_score.append(results_f1_score)
        

               
        #The next loop is used to extract the model after training in each node to do the averaging
        average_models=[]
        for idx, both_models in enumerate(all_models):
            #print(f"node {idx}")
            for idy, model in enumerate(both_models):
                if idy==1: #extract element in position 1
                    average_models.append(model.parameters())
                    
        
       #We average the models stored in average_models array       
        for param in start_model_general.parameters():
            count_number_nodes=0
            for element in average_models:
                
                if count_number_nodes==0:
                    a=next(element)
                else:
                    a=a+next(element)
                count_number_nodes=count_number_nodes+1
           
            average_param=a/count_number_nodes
          
            param.data=average_param
        print(start_model_general)
        round_current=round_current+1

        
        #MODEL GENERAL EVALUATION
        #This section is used to evaluate the general model which is the result of averaging the other nodes


        with torch.no_grad():            #We do not change the values of the model while we are evaluatins
            outputs = start_model_general(features_all_test_folds_to_evaluate_in_global)  #predict logits for each elemente in batch
            
            probabilities = nn.functional.softmax(outputs, dim=1) #use softmax since we do not use cross-entrophy
            _, predicted = torch.max(probabilities, 1)   #predict the probs
            
            pred_numpy=predicted.data.cpu().numpy()    #transform prediction to numpy from gpu
            labels_numpy=labels_all_test_folds_to_evaluate_in_global.data.cpu().numpy()  #transform labels to numpy from gpu
            
        
        cf_matrix = confusion_matrix(pred_numpy, labels_numpy)  #find confusion matrix
        target_names=['0','1','2','3','4','5','6','7','8','9']  
        #print(cf_matrix)
        report=classification_report(pred_numpy, labels_numpy,target_names=target_names,zero_division=0,output_dict=True)
     
        #metrics of general model evaluation    
        macro_f1 = report['macro avg']['f1-score']
        global_model_eval_macro.append(macro_f1 )
        class_0_f1 = report['0']['f1-score']
        global_model_eval_class_0.append(class_0_f1)
        class_1_f1 = report['1']['f1-score']
        global_model_eval_class_1.append(class_1_f1)
        class_2_f1 = report['2']['f1-score']
        global_model_eval_class_2.append(class_2_f1)
        class_3_f1 = report['3']['f1-score']
        global_model_eval_class_3.append(class_3_f1)
        class_4_f1 = report['4']['f1-score']
        global_model_eval_class_4.append(class_4_f1)
        class_5_f1 = report['5']['f1-score']
        global_model_eval_class_5.append(class_5_f1)
        class_6_f1 = report['6']['f1-score']
        global_model_eval_class_6.append(class_6_f1)
        class_7_f1 = report['7']['f1-score']
        global_model_eval_class_7.append(class_7_f1)
        class_8_f1 = report['8']['f1-score']
        global_model_eval_class_8.append(class_8_f1)
        class_9_f1 = report['9']['f1-score']
        global_model_eval_class_9.append(class_9_f1)



    #after training for many rounds we get the results
  
    print("results_f1_score")
    print( results_epoch_f1_score)
    epoch=[]
    classes={'macro':1,'number_0':2,'number_1':3,'number_2':4,'number_3':5,'number_4':6,'number_5':7,'number_6':8,'number_7':9,'number_8':10,'number_9':11,'Execution time':12}  #dict to iterate 
    classes_general_eval={'macro':global_model_eval_macro,'number_0':global_model_eval_class_0,'number_1':global_model_eval_class_1,'number_2':global_model_eval_class_2,'number_3':global_model_eval_class_3,'number_4':global_model_eval_class_4,'number_5':global_model_eval_class_5,'number_6':global_model_eval_class_6,'number_7':global_model_eval_class_7,'number_8':global_model_eval_class_8,'number_9':global_model_eval_class_9} #dict to iterate
    all_plots=[]
    

    for key, value in classes.items():  #iterate through the classes
        print(f"Key: {key}, Value: {value}")  
        model_general=[]
        node_1=[]
        node_2=[]
        node_3=[]
        node_4=[]
        node_5=[]
        epochs=0
        epochs_array=[]
        for result_epoch in results_epoch_f1_score:  #iterates through each epoch
            epochs=epochs+1
            for node in result_epoch:
                if node[0]=='node 1':
                    node_1.append(node[value])
                if node[0]=='node 2':
                    node_2.append(node[value])
                if node[0]=='node 3':
                    node_3.append(node[value])
                if node[0]=='node 4':
                    node_4.append(node[value])
                if node[0]=='node 5':
                    node_5.append(node[value])
            epochs_array.append(epochs)
        
        
        print("nodes")
        print(node_1)
        print(node_2)
        print(node_3)
        print(node_4)
        print(node_5)
        plt.figure()
        if key!='Execution time':   # conditional because classes_general_eval dict does not have execution_time
            plt.plot(epochs_array, classes_general_eval[key],label='node general')
        plt.plot(epochs_array, node_1,label='node 1')
        plt.plot(epochs_array, node_2,label='node 2')
        plt.plot(epochs_array, node_3,label='node 3')
        plt.plot(epochs_array, node_4,label='node 4')
        plt.plot(epochs_array, node_5,label='node 5')
        plt.legend()
        plt.xlabel('Epochs')
        if key!='Execution time':   #Change the ylabel in case of execution time
            plt.ylabel('F1-score')
        else:
            plt.ylabel('Seconds')

        plt.title(f'F1-score vs Epochs ({key})')
        plt.savefig(f'Figure class {key}.png')  # Save as PNG
        plt.close()  # Close the figure to release resources
  

   
    


   