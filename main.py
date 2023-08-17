import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing

# Define your model class
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    


          

# Number of models to train
num_models = 3

# Example datasets and targets for each model
datasets = [torch.randn(100, 10) for _ in range(num_models)]
targets = [torch.randn(100, 1) for _ in range(num_models)]

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
    model, data, target = args
    start_model_save_input.load_state_dict(model.state_dict())
   
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs=10
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

       #print local models
    print("AFTER TRAINING")
    for name, param in model.named_parameters():
       print(f"Name: {name}, Shape: {param}")
    

   
    both_models.append(start_model_save_input)
    both_models.append(model)

    return both_models

if __name__ == '__main__':
    # Combine models, datasets, and targets into a list of tuples
    print("INITIAL models ")
    i=0
    while(i<1):
        
        model1.load_state_dict(start_model_general.state_dict())
        model2.load_state_dict(start_model_general.state_dict())
        model3.load_state_dict(start_model_general.state_dict())
        # model4.load_state_dict(start_model_general.state_dict())
        # model5.load_state_dict(start_model_general.state_dict())
        #models=[model1,model2,model3,model4,model5]
        models=[model1,model2,model3]
        
        #create n copies of the general model depending on the number of models
        # models_general=[start_model_general]*num_models
        # models=models_general
        #print local models
        for idx, model in enumerate(models):
            print(f"Model {idx + 1} parameters:")
            for name, param in model.named_parameters():
                print(f"Name: {name}, Shape: {param}")
        
    

        
        #print general models. 
        print("model general")
    
        for name, param in start_model_general.named_parameters():
            print(f"Name: {name}, Shape: {param}")

        #pass parameters
        train_args = [(model, data, target) for  model, data, target in zip(models, datasets, targets)]

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
                    average_models.append(model)
                    
               
        print("Append model weights")
        print(average_models)

        for idx, model in enumerate(average_models):
            for (name, param) in (model.named_parameters()):                                        
                        print(f"Name: {name}, Shape: {param}")
                        #print(f"Name: {name}, Shape: {param.shape}")
               
    

        # print("ONE MODEL FINAL")
        # print(all_models[0][1])
        # start_model_general=all_models[0][1]
        # for name, param in start_model_general.named_parameters():
        #     print(f"Name: {name}, Shape: {param}")
        #     #print(f"Name: {name}, Shape: {param.shape}")
        # i=i+1
