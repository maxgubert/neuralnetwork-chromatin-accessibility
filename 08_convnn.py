import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)
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



# Define Neural Net

class ConvNet(torch.nn.Module):
    
    def __init__(self, input_len, num_channels, conv_kernel_size_nts, pool_kernel_size,
                 h1_size, flat_size, stride):

        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=num_channels,
                                     kernel_size=conv_kernel_size_nts*4, stride=stride)
        self.conv1_bn = torch.nn.BatchNorm1d(num_channels)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size, ceil_mode=True)
        self.h1 = torch.nn.Linear(flat_size, h1_size)
        self.dropout = torch.nn.Dropout(0.25)
        self.out = torch.nn.Linear(h1_size, 1)

    def forward(self, x):
        
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool1(self.conv1_bn(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.nn.functional.relu(self.h1(x))
        x = self.dropout(x)
        x = self.out(x)
        
        return x


# Parameters
input_len = X_train.shape[1]
num_channels = 10 
conv_kernel_size_nts = 10
pool_kernel_size = 2
h1_size = 1000
stride = 4
flat_size = ((1 + (input_len - (conv_kernel_size_nts*4)) // stride) // pool_kernel_size 
#+ 1
) * num_channels

epochs = 300
min_valid_loss = np.inf
patience = 3
trigger_times = 0

neural_net = ConvNet(input_len, num_channels, conv_kernel_size_nts, pool_kernel_size,
             h1_size, flat_size, stride).to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(neural_net.parameters(), lr = 0.0002, weight_decay=1e-5)



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
            print("Early stopping at epoch:", i + 1, "after", trigger_times, "epochs without improvement")
            break
        
    else:
        trigger_times = 0

    min_valid_loss = valid_loss
    
with torch.no_grad():              
    print("Evaluating model with test dataset")
    print("Test accuracy:", mean_squared_error(neural_net(X_test), y_test).item())


