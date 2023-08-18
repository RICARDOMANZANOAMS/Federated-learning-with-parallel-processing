import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
from sklearn.metrics import confusion_matrix
DIM_IN = 784
HIDDEN_SIZE = 100
DIM_OUT = 10
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

def preprocess_dataset(fold):
    columns_to_select_no_label=fold.columns[fold.columns!='label']
    
    features = fold.loc[:, columns_to_select_no_label]
    #print(features)
    labels=fold['label']
    #print(labels)

    features_np=features.to_numpy() #transform to numpy   
    features_t=torch.from_numpy(features_np).float()  #transform to torch
    
    labels_np=labels.to_numpy()   #transform to numpy labels
    labels_t=torch.tensor(labels_np)  # transform to torch label
    return features_t,labels_t


# Number of models to train
num_models = 5

# Create a list of models
seed=20   
torch.manual_seed(seed)    #specify the seed to start weigths in the neural network
# models = [Classifier() for _ in range(num_models)]
#print model param

start_model_general=Classifier() 
model1=Classifier() 
model2=Classifier() 
model3=Classifier()
model4=Classifier()
model5=Classifier()


start_model_save_input=Classifier()
# Define the worker function for training
def train_worker(args):
    both_models=[]
    model, data, target,data_test,target_test= args
    start_model_save_input.load_state_dict(model.state_dict())
   
    

    criterion= nn.CrossEntropyLoss()
    
    optimizer= torch.optim.Adam(model.parameters(), lr=1e-4)


    epochs=10
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

 
       #print local models
    with torch.no_grad():            #no train the model with test 
        outputs = model(data_test)  #predict logits for each elemente in batch
        
        probabilities = nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
        
        pred_numpy=predicted.data.cpu().numpy()    #transform prediction to numpy from gpu
        labels_numpy=target_test.data.cpu().numpy()  #transform labels to numpy from gpu
        
       
    cf_matrix = confusion_matrix(pred_numpy, labels_numpy)  #find confusion matrix
    target_names=['0','1','2','3','4','5','6','7','8','9']  
    print(cf_matrix)
    # report=classification_report(y_true, y_pred,target_names=target_names,zero_division=0,output_dict=True)
    # print(report)
    # macro_f1 = report['macro avg']['f1-score']
    # print("macro")
    # print(macro_f1)
    


    both_models.append(start_model_save_input)
    both_models.append(model)

    return both_models

if __name__ == '__main__':
    # Combine models, datasets, and targets into a list of tuples
    print("INITIAL models ")
    import pandas as pd

    fold1_train=pd.read_csv('C:/RICARDO/2023 CISTECH BACKUP/cistech/2023/MNIST DATASET FEDERATED IN ONE COMPUTER/test federated learning mnist/Mnist dataset zero day attack/Dataset 5 folds no two classes in each fold/fold1_train_mnist_5_folds.csv')
    fold1_train=fold1_train.iloc[:,1:]

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

    folds_train_features=[features_fold1_train,features_fold2_train,features_fold3_train,features_fold4_train,features_fold5_train]
    folds_train_labels=[labels_fold1_train,labels_fold2_train,labels_fold3_train,labels_fold4_train,labels_fold5_train]

    folds_test_features=[features_fold1_test,features_fold2_test,features_fold3_test,features_fold4_test,features_fold5_test]
    folds_test_labels=[labels_fold1_test,labels_fold2_test,labels_fold3_test,labels_fold4_test,labels_fold5_test]
    i=0

    while(i<3):
        
        model1.load_state_dict(start_model_general.state_dict())
        model2.load_state_dict(start_model_general.state_dict())
        model3.load_state_dict(start_model_general.state_dict())
        model4.load_state_dict(start_model_general.state_dict())
        model5.load_state_dict(start_model_general.state_dict())
        models=[model1,model2,model3,model4,model5]
       
        
          

        
        #print general models. 
        print("model general")
    
        for name, param in start_model_general.named_parameters():
            print(f"Name: {name}, Shape: {param}")

        #pass parameters
        train_args = [(model, data, target,data_test,target_test) for  model, data, target,data_test,target_test in zip(models, folds_train_features, folds_train_labels,folds_test_features, folds_test_labels)]

        # Create a multiprocessing pool
        with multiprocessing.Pool(processes=num_models) as pool:
            # Train models concurrently and collect results
            all_models = pool.map(train_worker, train_args)
            pool.close()
            pool.join()
    

        print("RICARDOOOOOOOOOOOOOOOOOOO")
        print("trained models")
        # Now, trained_models contains the trained models
        for idx, both_models in enumerate(all_models):
            print(f"node {idx}")
            for model in both_models:
                for name, param in model.named_parameters():
                    print(f"Name: {name}, Shape: {param}")
                    #print(f"Name: {name}, Shape: {param.shape}")
                print("=" * 20)

        average_models=[]
        for idx, both_models in enumerate(all_models):
            print(f"node {idx}")
            for idy, model in enumerate(both_models):
                if idy==1:
                    average_models.append(model.parameters())
                    
               
        print("Append model weights")
        print(average_models)

        
                      
                        
                        #print(f"Name: {name}, Shape: {param.shape}")
               
      
     



        
        for param in start_model_general.parameters():
            count_number_nodes=0
            for element in average_models:
                
                if count_number_nodes==0:
                    a=next(element)
                else:
                    a=a+next(element)
                count_number_nodes=count_number_nodes+1
            # print(a)
            average_param=a/count_number_nodes
            # print("average")
            # print(average_param)
            param.data=average_param
        print(start_model_general)
        i=i+1
        for (name, param) in (start_model_general.named_parameters()):      
                                                          
            print(f"Name: {name}, Shape: {param}")
        print("--------------------new-------------------------")
        # i=i+1
        # print("model concat")
        # for (name, param) in (start_model_general.named_parameters()):                                        
        #     print(f"Name: {name}, Shape: {param}")
            #print(f"Name: {name}, Shape: {param.shape}")

        # print("ONE MODEL FINAL")
        # print(all_models[0][1])
        # start_model_general=all_models[0][1]
        # for name, param in start_model_general.named_parameters():
        #     print(f"Name: {name}, Shape: {param}")
        #     #print(f"Name: {name}, Shape: {param.shape}")
        # i=i+1
