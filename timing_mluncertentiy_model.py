# -*- coding: utf-8 -*-
"""Timing_MLUncertentiy_Model.ipynb """

# Imports -------------------------------------------------------------
import h5py
import numpy as np
import torch
from torch import nn
from torch import from_numpy
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
import torch.nn.parallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm
from scipy.sparse import lil_matrix
from sys import getsizeof
import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.sparse import lil_matrix
import time

# Defining the PyTorch model -------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, input_size, output_size, num_layers,width,activ = "ReLU",test_std = 0,uniform=False):
        super(Model, self).__init__()
        self.relustack = OrderedDict()
        self.flatten = nn.Flatten()

        if test_std != None:
            stand_dev = test_std
        else:
            if activ == "ReLU":
                stand_dev = 2
            elif activ == "Tanh":
                stand_dev = 1
            elif activ == "GELU":
                stand_dev = 2
            else:
                print("NO ACTIVATION FUNC SET")

        for i in range(num_layers):
            if i == 0:
                self.relustack["linear1"] = nn.Linear(input_size, width)
                if activ == "ReLU":
                    self.relustack["relu1"] = nn.ReLU()
                elif activ == "Tanh":
                    self.relustack["relu1"] = nn.Tanh()
                elif activ == "GELU":
                    self.relustack["relu1"] = nn.GELU()
                else:
                    print("NO ACTIVATION FUNC SET")
            elif i==num_layers - 1:
                self.relustack["linear"+str(i+1)] = nn.Linear(width, output_size)
            else:
                self.relustack["drop"+str(i+1)] = nn.Dropout(0.04)
                self.relustack["linear"+str(i+1)] = nn.Linear(width, width)
                if activ == "ReLU":
                    self.relustack["relu"+str(i+1)] = nn.ReLU()
                elif activ == "Tanh":
                    self.relustack["relu"+str(i+1)] = nn.Tanh()
                elif activ == "GELU":
                    self.relustack["relu"+str(i+1)] = nn.GELU()

        self.linear_relu_stack = nn.Sequential(self.relustack)

        if not uniform:
            for layer in self.linear_relu_stack:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight,std=np.sqrt((stand_dev/float(width))))
                    layer.bias.data.fill_(0)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Custom dataset class -------------------------------------------------------------
class CalorimeterDataset(Dataset):
    def __init__(self, file_path, output_vars, input_vars, lower, higher, stand=False,mult = 1,split=False,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),conv=False,not_feature_input_vars=None,split_perc=.8):
        self.file_path = file_path
        self.num_columns = 0
        self.mult = mult
        self.output_vars = output_vars
        self.input_vars = input_vars
        self.split = split
        self.num_list = [str(i) for i in range(lower,higher)]
        self.conv = conv
        self.split_perc = split_perc

        self.getFilelist()
        old_len = 0

        if not conv:
            for h5_file in self.file_list:

                samples_temp = h5py.File(h5_file, 'r')
                samples = {}

                for key in dict(samples_temp).keys():
                    if key in input_vars or key in output_vars:
                        if key == "ECAL":
                            samples[key] = lil_matrix(samples_temp.get(key)[()][:,0:25,0:25,:].reshape(samples_temp.get(key)[()].shape[0], -1))
                        else:
                            samples[key] = lil_matrix(samples_temp.get(key)[()].reshape(samples_temp.get(key)[()].shape[0], -1))

                del samples_temp

                cnt = 1
                for key in input_vars:
                    self.data[old_len:old_len+(samples[key]).shape[0], cnt:cnt+self.size_dict[key]] = samples[key]
                    cnt=cnt+self.size_dict[key]-1

                cnt = 1
                for key in output_vars:
                    self.data[old_len:old_len+(samples[key]).shape[0], -cnt] = samples[key]
                    cnt=cnt+1

                old_len = old_len + samples[input_vars[0]].shape[0]
                del samples

            # if stand:
            #     self.normalize_2d()
            self.test_train_split()

        else:
            if not_feature_input_vars:
                self.data = torch.zeros(self.total_len,135,25,25)
            else:
                self.data = torch.zeros(self.total_len,86,25,25)

            for h5_file in self.file_list:

                samples_temp = h5py.File(h5_file, 'r')
                samples = {}

                for key in dict(samples_temp).keys():
                    if key in input_vars or key in output_vars:
                        if key == "ECAL":
                            samples[key] = from_numpy(samples_temp.get(key)[()][:,0:25,0:25,:])
                        else:
                            samples[key] = from_numpy(samples_temp.get(key)[()])

                    if key not in not_feature_input_vars:
                        samples[key] = from_numpy(samples_temp.get(key)[()])

                del samples_temp

                for key in input_vars:
                    if key == "ECAL":
                        self.data[old_len:old_len+(samples[key]).shape[0], 0:25,0:25,0:25] = samples[key]
                    elif key == "HCAL":
                        self.data[old_len:old_len+(samples[key]).shape[0], 25:85,0:11,0:11] = samples[key].permute(0,3, 1, 2)

                if not_feature_input_vars:
                    cnt = 0
                    for key in dict(samples).keys():
                        if key not in not_feature_input_vars:
                            self.data[old_len:old_len+(samples[key]).shape[0],85+cnt,0,0] = samples[key]
                            cnt +=1

                cnt = 1
                for key in output_vars:
                    self.data[old_len:old_len+(samples[key]).shape[0],-1,0,0,] = samples[key]
                    cnt=cnt+1

                old_len = old_len + samples[input_vars[0]].shape[0]
                del samples

            self.data.to_sparse()
            self.test_train_split()

    def test_train_split(self):

        if not self.conv:
            output_size = len(self.output_vars)
            x_ids = list(range(self.data.shape[0]))
            x_train_ids, x_test_ids, Y_train, self.y_test = train_test_split(x_ids,self.data[:, -output_size:],test_size=1-self.split_perc, train_size=self.split_perc)
            del Y_train

            self.X_test = self.data[x_test_ids,:-output_size]
            self.data = self.data[x_train_ids,:]

            del x_train_ids
            del x_test_ids
        else:
            output_size = len(self.output_vars)
            x_ids = list(range(self.data.shape[0]))
            x_train_ids, x_test_ids, Y_train, self.y_test = train_test_split(x_ids,self.data[:,-1,0,0],test_size=1-self.split_perc, train_size=self.split_perc)
            del Y_train

            self.X_test = self.data[x_test_ids,:-1,:,:]
            self.data = self.data[x_train_ids,:,:,:]

            del x_train_ids
            del x_test_ids

    def getFilelist(self):
        self.total_len = 0
        max_energy = 0
        min_energy = 100

        if not os.path.isdir(self.file_path):
            self.file_list = [self.file_path]
            samples = h5py.File(self.file_path, 'r')
            self.total_len = self.total_len + int(len(samples.get(self.output_vars[0])[()]))

            # finding energies
            e_index = self.file_path.find("/e_")
            underscore_index = self.file_path.find("_", e_index + 3 )
            max_energy = int(self.file_path[e_index + 3:underscore_index])
            min_energy = max_energy

        else:
            self.file_list = []
            cnt_lim = 10000000000
            cnt = 0
            # Get all files in the directory with the specified extension
            for root, dirs, files in os.walk(self.file_path):
                for file in files:
                    if file.endswith(".h5") and cnt<cnt_lim and "e_" in str(file) and any(str("_")+i+"." in str(file) for i in self.num_list):

                        file_name = str(file)
                        e_index = file_name.find("e_")
                        underscore_index = file_name.find("_", e_index + 2 )
                        num = float(file_name[e_index + 2:underscore_index])

                        if num > 0:
                            cnt = cnt + 1

                            self.file_list.append(os.path.join(root, file))

                            samples = h5py.File(os.path.join(root, file), 'r')
                            self.total_len = self.total_len + int(len(samples.get(self.output_vars[0])[()]))

                            if num<min_energy:
                                min_energy = num
                            if num>max_energy:
                                max_energy = num

        self.size_dict = {}
        for input_var in self.input_vars:
            if input_var == "ECAL":
                size_output = 25*25*25
                self.num_columns = self.num_columns + size_output
                self.size_dict[input_var] = size_output
            else:
                size_output = int(np.shape(samples.get(input_var)[()])[1]*np.shape(samples.get(input_var)[()])[2]*np.shape(samples.get(input_var)[()])[3])
                self.num_columns = self.num_columns + size_output
                self.size_dict[input_var] = size_output

        del samples
        self.data = lil_matrix((self.total_len,self.num_columns),dtype=float)
        print("Training with "+str(int(self.total_len*self.split_perc))+" samples")
        print("Testing with "+str(int(self.total_len*(1-self.split_perc)))+" samples")
        print("Samples are from energy "+str(min_energy)+" GeV to energy "+str(max_energy)+" GeV")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        if self.conv:
            return (self.data[idx,:,:,:])
        else:
            return (self.data[idx,:].toarray())[0,:]

    def getNumColumn(self):
        return self.num_columns

    def getTestdata(self):
        num_rows = self.X_test.shape[0]
        rows_per_piece = num_rows
        num_pieces = num_rows // rows_per_piece
        file_dict = {}
        file_dict['X'] = []
        file_dict['Y'] = []

        pieces_x = []
        pieces_y = []
        start_row = 0
        for i in range(num_pieces - 1):
            end_row = start_row + rows_per_piece

            if not self.conv:
                piece = self.X_test[start_row:end_row, :]
                pieces_x.append(piece)
                piece = self.y_test[start_row:end_row, :]
                pieces_y.append(piece)
                start_row = end_row
            else:
                piece = self.X_test[start_row:end_row, :,:,0:85]
                pieces_x.append(piece)
                piece = self.y_test[start_row:end_row]
                pieces_y.append(piece)
                start_row = end_row

        if not self.conv:
            pieces_x.append(self.X_test[start_row:, :])  # Last piece
            pieces_y.append(self.y_test[start_row:, :])  # Last piece
        else:
            pieces_x.append(self.X_test[start_row:, :,:,0:85])  # Last piece
            pieces_y.append(self.y_test[start_row:])  # Last piece

        for i, piece in enumerate(pieces_x):
            file_name = "test_data/piece_x_"+str(i+1)+".npy"
            if self.conv:
                np.save(file_name, piece.numpy())
            else:
                np.save(file_name, piece.toarray())
            file_dict['X'].append(file_name)

        for i, piece in enumerate(pieces_y):
            file_name = "test_data/piece_y_"+str(i+1)+".npy"
            if self.conv:
                np.save(file_name, piece.numpy())
            else:
                np.save(file_name, piece.toarray())

            file_dict['Y'].append(file_name)

        return file_dict


time_orgin = time.perf_counter()
# Defining the dataset -------------------------------------------------------------
# what inputs to use and what to regress to
input_vars = ["HCAL","ECAL"]
output_vars = ['energy']

# where the dataset is located
file_path = "/Users/ibrahim/Benchmarks/data" # You may have to change this path - looking like it must be on the drive of the person who runs

# creating the dataset class - the number of files used is tied to what energy range
dataset_1 = CalorimeterDataset(file_path, output_vars,input_vars,10,30,stand=False) # 10-12 GeV Test
batch_size = len(dataset_1) # For Full Batch
dataloader = DataLoader(dataset_1, batch_size=batch_size, shuffle=True)
time_end = time.perf_counter()
print("Computer took "+ str(time_end-time_orgin)+" seconds to load data set")

# Defining Model Hyperparameters -------------------------------------------------------------
learing_rate = .0006
width = 64
depth = 4
num_epochs = 50
# GPU device
device = torch.device("mps")
print(device)
# Model Variables
output_size = int(len(output_vars))
input_size = dataset_1.getNumColumn() - len(output_vars)

# Defining the model -------------------------------------------------------------
model = Model(input_size, output_size,depth,width,activ="Tanh",uniform=False)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learing_rate)

# Training the model -------------------------------------------------------------
print("Training Network with " +str(depth)+ " layers")

# setting training variables
time_orgin = time.perf_counter()
time_last = time_orgin
running_loss_vec = []
timing_vec = []
epoch_vec = list(range(num_epochs))

for epoch in tqdm(range(num_epochs), desc='Progress', unit='epochs'):
    for data in dataloader:
        ### Training model
        inputs = data[:, :-output_size].float().to(device)
        targets = data[:, -output_size:].float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs.float(), targets.float())
        loss.backward()
        optimizer.step()

    running_loss_vec.append(loss.item())
    time_epoch = time.perf_counter()
    timing_vec.append(time_epoch-time_last)
    time_last = time_epoch

time_end = time.perf_counter()

print("Trained Network with "+str(depth)+" layers and "+str(width)+" width and found an MSE of " + str(np.min(running_loss_vec)))
print("Network took "+ str(time_end-time_orgin)+" seconds to Train")

# Commented out IPython magic to ensure Python compatibility.
# Plotting Results -------------------------------------------------------------
# %matplotlib inline
fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
if num_epochs > 100:
  ax.set_yscale("log")
ax.set_ylabel('Loss function Value (GeV)')
plt.title("Epoch vs Loss Function for r= ("+str(depth)+","+str(width)+")")
ax.plot(epoch_vec, running_loss_vec,label="MSE on Training Data",color = 'm')
ax.legend()
plt.savefig("loss_plot.png")

fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
ax.set_ylabel('Seconds')
plt.title("Epoch vs Timing for r= ("+str(depth)+","+str(width)+")")
ax.plot(epoch_vec, timing_vec,label="Number of Seconds Per Epoch",color = 'm')
ax.legend()
plt.savefig("epoch_plot.png")

