import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim as optim

data = pd.read_csv('./coin_Bitcoin.csv')

# take a look at the csv file yourself first
# columns High, Low, Open are input features and column Close is target value
x = data[['High', 'Low', 'Open']].values
y = data['Close'].values

scaler_x = StandardScaler()
scaler_y = StandardScaler()

# use StandardScaler from sklearn to standardize
x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform((y.reshape(-1, 1)))


# split into train and evaluation (8 : 2) using train_test_split from sklearn
train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.2, random_state=100)


# now make x and y tensors, think about the shape of train_x, it should be (total_examples, sequence_lenth, feature_size)
# we wlll make sequence_length just 1 for simplicity, and you could use unsqueeze at dimension 1 to do this
# also when you create tensor, it needs to be float type since pytorch training do not take default type read using pandas
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_x = torch.tensor(test_x, dtype=torch.float32) 
train_x = train_x.unsqueeze(1)
test_x = test_x.unsqueeze(1)
seq_len = train_x[0].shape[0] # it is actually just 1 as explained above


# different from CNN which uses ImageFolder method, we don't have such method for RNN, so we need to write dataset class ourselves, reference tutorial is in main documentation
class BitCoinDataSet(Dataset):
    def __init__(self, train_x, train_y):
        super(Dataset, self).__init__()
        self.train_x = train_x
        self.train_y = train_y
        

    def __len__(self):
        return len(self.train_x)
        

    def __getitem__(self, idx):
        x = self.train_x[idx]
        y = self.train_y[idx]
        return x, y
        


# now prepare dataloader for training set and evaluation set, and hyperparameters
hidden_size =128
num_layers = 2
learning_rate =0.001
batch_size = 64
epoch_size = 10

train_dataset = BitCoinDataSet(train_x, train_y)
test_dataset = BitCoinDataSet(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#import torch.nn as nn
# model design goes here
class RNN(nn.Module):

    # there is no "correct" RNN model architecture for this lab either, you can start with a naive model as follows:
    # lstm with 5 layers (or rnn, or gru) -> linear -> relu -> linear
    # lstm: nn.LSTM (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

    def __init__(self, input_feature_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_feature_size, hidden_size, num_layers, batch_first=True)

        # Define linear layers
        self.fc1 = nn.Linear(hidden_size, 64)  # Adjust the output size as needed
        self.fc2 = nn.Linear(64, 1)  # Output size 1 for regression task

        # Define activation function (e.g., ReLU)
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        # Forward pass through LSTM
        out, _ = self.lstm(x)

        # Take the output from the last time step
        out = out[:, -1, :]

        # Apply linear layers with ReLU activation
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out
        


# instantiate your rnn model and move to device as in cnn section
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rnn = RNN(x.shape[1], hidden_size, num_layers).to(device)
# loss function is nn.MSELoss since it is regression task
criteria = nn.MSELoss()
# yo ucan start with using Adam as optimizer as well 
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)


criterion = nn.MSELoss()
# start training 
rnn.train()
for epoch in range(epoch_size): # start with 10 epochs

    running_loss = 0.0 # you can print out average loss per batch every certain batches

    for batch_idx, data in enumerate(train_loader):
        # get inputs and target values from dataloaders and move to device
        inputs, targets = data
        inputs.to(device)
        targets.to(device)

        # zero the parameter gradients using zero_grad()
        optimizer.zero_grad()

        # forward -> compute loss -> backward propogation -> optimize (see tutorial mentioned in main documentation)
        outputs = rnn(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() # add loss for current batch
        if batch_idx % 100 == 99:    # print average loss per batch every 100 batches
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')



prediction = []
ground_truth = []
# evaluation
rnn.eval()
with torch.no_grad():
    for data in test_loader:
        inputs, targets = data
        inputs = inputs.to(device)

        ground_truth += targets.flatten().tolist()
        out = rnn(inputs).detach().cpu().flatten().tolist()
        prediction += out


# remember we standardized the y value before, so we must reverse the normalization before we compute r2score
prediction = scaler_y.inverse_transform([prediction])
ground_truth = scaler_y.inverse_transform([ground_truth])
prediction = prediction[0]
ground_truth = ground_truth[0]
# use r2_score from sklearn
r2score = r2_score(prediction,ground_truth)
print('r2score',r2score)
