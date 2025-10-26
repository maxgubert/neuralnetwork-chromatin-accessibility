import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", device)


#Load data
def loadData(tsv_fn):
    data = pd.read_csv(tsv_fn, sep = "\t")
    seq = data.iloc[:,4:808].values
    values = data.iloc[:,808:809].values
    values_arcsinh = np.arcsinh(values)
    data_tensors = torch.tensor(seq, dtype=torch.float64).to(device), torch.tensor(values_arcsinh, dtype=torch.float64).to(device)

    return data_tensors


X_train, y_train = loadData("~/nnPib2021/group_MM/Splits/C02M02/file_train_u.tsv")
X_val, y_val = loadData("~/nnPib2021/group_MM/Splits/C02M02/file_val_u.tsv")
X_test, y_test = loadData("~/nnPib2021/group_MM/Splits/C02M02/file_test_u.tsv")



# Build class Dataset
class Dataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
    
training_data = Dataset(X_train, y_train)
validation_data = Dataset(X_val, y_val)
    
training_batch = DataLoader(dataset=training_data, batch_size=32, shuffle=True)
validation_batch = DataLoader(dataset=validation_data, batch_size=32, shuffle=True)



# Define NN
class NeuralNet(torch.nn.Module):
    
    def __init__(self, input_len, h1_size):

        super(NeuralNet, self).__init__()
        
        self.h1 = torch.nn.Linear(input_len, h1_size)   # Hidden layer
        self.h1_bn = torch.nn.BatchNorm1d(h1_size)      # Batch normalization
        self.dropout = torch.nn.Dropout(0.25)           # Dropout
        self.output_layer = torch.nn.Linear(h1_size, 1) # Output layer

    def forward(self, x):
        
        x = self.h1(x)
        x = torch.nn.functional.relu(self.h1_bn(x))     # Activation hidden layer with ReLU
        x = self.dropout(x)                             # Dropout
        x = self.output_layer(x)                        # Activation output layer
       
        return x



# Parameters
input_len = X_train.shape[1]
h1_size = 2000
epochs = 50
min_valid_loss = np.inf
patience = 3
trigger_times = 0

neural_net = NeuralNet(input_len, h1_size).to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(neural_net.parameters(), lr = 0.0002, weight_decay=0)



# Training NN
for i in range(epochs):
    train_loss = 0
    neural_net.train()
    for x, y in training_batch:
        optimizer.zero_grad()
        pred = neural_net(x)
        output = loss_function(pred, y)
        output.backward()
        optimizer.step()
        train_loss += output.item()
    
    with torch.no_grad():
        valid_loss = 0
        neural_net.eval()
        for x, y in validation_batch:
            pred = neural_net(x)
            output = loss_function(pred, y)
            valid_loss += output.item()

    print("Epoch:", i+1, "Training loss:", train_loss/len(training_batch), "Validation loss:", valid_loss/len(validation_batch))
    
    if min_valid_loss < valid_loss:
        trigger_times += 1
        #print("Trigger times:", trigger_times)
        if trigger_times >= patience:
            print("Early stopping at epoch:", i, "after", trigger_times, "epochs without improvement")
            break
        
    else:
        trigger_times = 0

    min_valid_loss = valid_loss
    
with torch.no_grad():              
    print("Evaluating model with test dataset")
    print("Test accuracy:", mean_squared_error(neural_net(X_test), y_test).item())
        

# Plot loss
losses_df = pd.DataFrame({'epoch': np.arange(1, epoch_stop + 1), 'train_loss': train_MSE, 'val_loss': val_MSE})   
    
plt.figure()
sns.lineplot(x = "epoch", y = "train_loss", data=losses_df, label="Train")
sns.lineplot(x = "epoch", y = "val_loss", data=losses_df, label="Validation")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Model loss")
plt.savefig("Model_loss_simple_opt.png")  


