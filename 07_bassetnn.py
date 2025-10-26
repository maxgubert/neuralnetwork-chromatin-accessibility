import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
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
class NN_Basset(torch.nn.Module):
    
    def __init__(self,
                 input_len,
                 c1_num_channels,
                 c1_conv_kernel_size_nts,
                 c1_stride,
                 c1_pool_kernel_size,
                 c2_num_channels,
                 c2_conv_kernel_size_nts,
                 c2_stride,
                 c2_pool_kernel_size,
                 c3_num_channels,
                 c3_conv_kernel_size_nts,
                 c3_stride,
                 c3_pool_kernel_size,
                 fc1_size,
                 fc2_size):
        
        super(NN_Basset, self).__init__()
               
        # Conv1
        self.conv1 = torch.nn.Conv1d(in_channels=1,
                                     out_channels=c1_num_channels,
                                     kernel_size=c1_conv_kernel_size_nts * 4,
                                     stride=c1_stride) 
        self.conv1_bn = torch.nn.BatchNorm1d(c1_num_channels)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool1d(kernel_size=c1_pool_kernel_size,
                                        stride=c1_pool_kernel_size,
                                        ceil_mode=True)
        
        # Conv2
        self.conv2 = torch.nn.Conv1d(in_channels=c1_num_channels,
                                     out_channels=c2_num_channels,
                                     kernel_size=c2_conv_kernel_size_nts,
                                     stride=c2_stride)
        self.conv2_bn = torch.nn.BatchNorm1d(c2_num_channels)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool1d(kernel_size=c2_pool_kernel_size,
                                        stride=c2_pool_kernel_size,
                                        ceil_mode=True)
        
        # Conv3
        self.conv3 = torch.nn.Conv1d(in_channels=c2_num_channels,
                                     out_channels=c3_num_channels,
                                     kernel_size=c3_conv_kernel_size_nts,
                                     stride=c3_stride)
        self.conv3_bn = torch.nn.BatchNorm1d(c3_num_channels)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool1d(kernel_size=c3_pool_kernel_size,
                                        stride=c3_pool_kernel_size,
                                        ceil_mode=True)
        
        # Full1
        self.fc1 = torch.nn.Linear(400, fc1_size)
        self.relu1_fc = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.3)
        
        # Full2
        self.fc2 = torch.nn.Linear(fc1_size, fc2_size)
        self.relu2_fc = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.3)
        
        # Out
        self.out = torch.nn.Linear(fc2_size, 1)


    def forward(self, x):
        # Add 1 extra dimension
        x = x.unsqueeze(1)                  # [16, 1, 804]
        # Conv1
        x = self.conv1(x)                   # [16, 300, 183]
        x = self.relu1(self.conv1_bn(x)) 
        x = self.pool1(x)                   # [16, 300, 61] 
        # Conv2
        x = self.conv2(x)                   # [16, 200, 51]
        x = self.relu2(self.conv2_bn(x))
        x = self.pool2(x)                   # [16, 200, 13]
        # Conv3
        x = self.conv3(x)                   # [16, 200, 7]
        x = self.relu3(self.conv3_bn(x))
        x = self.pool3(x)                   # [16, 200, 2]
        # Flattening
        x = torch.flatten(x, start_dim=1)   # [16, 400]
        # Full connected 1
        x = self.fc1(x)                     # [16, 1000]
        x = self.relu1_fc(x)
        x = self.dropout1(x)
        # Full connected 2
        x = self.fc2(x)                     # [16, 1000]
        x = self.relu2_fc(x)
        x = self.dropout2(x)
        # Output layer
        x = self.out(x)                     # [16, 1]
        
        return x


# Parameters
input_len = X_train.shape[1]
c1_num_channels = 300
c1_conv_kernel_size_nts = 19
c1_stride = 4
c1_pool_kernel_size = 3
c2_num_channels = 200
c2_conv_kernel_size_nts = 11
c2_stride = 1
c2_pool_kernel_size = 4
c3_num_channels = 200
c3_conv_kernel_size_nts = 7
c3_stride = 1
c3_pool_kernel_size = 4
fc1_size = 1000
fc2_size = 1000


epochs = 50
min_valid_loss = np.inf
patience = 3
trigger_times = 0

neural_net = NN_Basset(input_len, c1_num_channels, c1_conv_kernel_size_nts, c1_stride, c1_pool_kernel_size,
                       c2_num_channels, c2_conv_kernel_size_nts, c2_stride, c2_pool_kernel_size, c3_num_channels,
                       c3_conv_kernel_size_nts, c3_stride, c3_pool_kernel_size, fc1_size, fc2_size).to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(neural_net.parameters(), lr = 0.002, weight_decay=1e-4)

train_MSE = []
val_MSE = []
epoch_stop = epochs



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
    
    train_MSE.append(train_loss/len(training_batch))
    val_MSE.append(valid_loss/len(validation_batch))
    
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
    
    
    
# Plot loss
losses_df = pd.DataFrame({'epoch': np.arange(1, epoch_stop + 1), 'train_loss': train_MSE, 'val_loss': val_MSE})   
    
plt.figure()
sns.lineplot(x = "epoch", y = "train_loss", data=losses_df, label="Train")
sns.lineplot(x = "epoch", y = "val_loss", data=losses_df, label="Validation")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Model loss")
plt.savefig("Model_loss_Basset.png")  