#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:27:42 2024

@author: natasasamardzic
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import itertools
import matplotlib.pyplot as plt

directory = '/Users/jovanazoranovic/Documents/Doktorske/NATASA'

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# Random Matrix
def random_feedback_matrix(in_dim, out_dim):
    return torch.randn(out_dim, in_dim, device=device, requires_grad=False)


class MIF(nn.Module):
   
     def __init__(self, R_on=10, R_off=10e3, tau_alpha=100, C=100e-6, uv=1e-10, D=10e-9, Rx=1, Cx=100, N=2):
        super(MIF, self).__init__()
        self.R_on = R_on
        self.R_off = R_off
        self.tau_alpha = tau_alpha
        self.C = C
        self.uv = uv
        self.D = D
        self.Rx = Rx
        self.Cx = Cx
        self.N = N

     def forward(self, _input, x, Gm, a, I, v):
        a = -a / self.tau_alpha + _input
        I = (a - I) / self.tau_alpha + I
        v = (I - Gm * v) / self.C + v
       # x = -x / (self.Rx * self.Cx) + I * self.uv * self.R_on * (1 - (2 * x - 1)**2 / (1 - (2 * x - 1)**2 + (2 * x - 1)**(2 * self.N))) / ((self.D**2) * self.Cx)
        x = -x / (self.Rx * self.Cx) + v *Gm * self.uv * self.R_on * (1 - (2 * x - 1)**2 / (1 - (2 * x - 1)**2 + (2 * x - 1)**(2 * self.N))) / ((self.D**2) * self.Cx)+x 
        x = torch.clamp(x, 0, 1)
        Gm = 1 / (x * self.R_on + (1 - x) * self.R_off)
        return x, Gm, a, I, v

     def init_MIF(self, batch_size, *args):
        
        x = torch.ones((batch_size, *args), device=device, dtype=dtype) * 0.000000001
      
        Gm = 1 / (x * self.R_on + (1 - x) * self.R_off)
        v = torch.ones((batch_size, *args), device=device, dtype=dtype)
        I = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        a = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        return x, Gm, a, I, v
    

num_inputs = 28*28
num_hidden = 100
num_outputs = 10    
B1 = random_feedback_matrix(num_outputs, num_hidden)
#B2 = random_feedback_matrix(num_outputs, num_inputs)


batch_size = 200
num_epochs = 50
data_path='/tmp/data/mnist'



num_steps = 1000
#num_steps = 25
#beta = 0.95


class Net_DFA(nn.Module):
    def __init__(self):
        super(Net_DFA, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.relu2 = nn.ReLU()
        self.mif2 = MIF()

    def forward(self, x):
        x_out, Gm_out, a_out, I_out, v_out = self.mif2.init_MIF(batch_size, num_outputs)
        v_rec = []
        hidden_rec = []
        inp2 = torch.zeros((1, num_outputs)).to(device)

        for step in range(num_steps):
            out1 = self.fc1(x)
            out1 = self.relu1(out1)
            hidden_rec.append(out1)
            out2 = self.fc2(out1)
            out2 = self.relu2(out2)

            if step in [0, 400, 800]:
                x_out, Gm_out, a_out, I_out, v_out = self.mif2(out2, x_out, Gm_out, a_out, I_out, v_out)
            else:
                x_out, Gm_out, a_out, I_out, v_out = self.mif2(inp2, x_out, Gm_out, a_out, I_out, v_out)

            v_rec.append(v_out)

        return torch.stack(v_rec, dim=0), torch.stack(hidden_rec, dim=0)



    # preprocesing
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])


mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)


# DataLoader
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)





net = Net_DFA().to(device)

log_softmax_fn=nn.LogSoftmax(dim=-1)
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


train_loss_hist = []
test_loss_hist = []
train_acc_hist = []
test_acc_hist = []
counter = 0

# Trening  DFA

for epoch in range(num_epochs):
    minibatch_counter = 0

    for data_it, targets_it in train_loader:
        data_it = data_it.to(device)
        targets_it = targets_it.to(device)

        # Forward pass
        v_rec, hidden_output = net(data_it.view(batch_size, -1))

      
        if v_rec.dim() == 3:  
            v_rec = v_rec.mean(dim=0)  # average over timesteps
        elif v_rec.dim() == 2:  
            pass
        else:
            raise ValueError(f"v_rec shape: {v_rec.shape}")

       
        if targets_it.dtype != torch.int64:
            targets_it = targets_it.to(torch.int64)  

       

        # logprobas and loss
        log_probs = log_softmax_fn(v_rec)
        loss_val = loss_fn(log_probs, targets_it)

        #er 
        output_error = log_probs.exp()
        output_error[range(batch_size), targets_it] -= 1
        output_error /= batch_size

      
        for param in net.parameters():
            param.grad = None

     
        with torch.no_grad():
            
            data_flat = data_it.view(batch_size, -1)

           #h1
            a1=net.fc1(data_flat)
            da1=(a1 > 0).to(a1.dtype)
            hidden_out = net.relu1(net.fc1(data_flat))

            # dW2 db2
            net.fc2.weight.grad = torch.matmul(output_error.T, hidden_out) / batch_size
            net.fc2.bias.grad = output_error.sum(dim=0) / batch_size

          
           # print(f"Output error : {output_error.shape}")
           # print(f" B1: {B1.shape}")
           # dout1
            #hidden_error = torch.matmul(output_error, B1.T)  #  (batch_size, num_hidden)
            hidden_error=(output_error@B1.T)*da1
           #dW1 db1
            net.fc1.weight.grad = torch.matmul(hidden_error.T, data_flat) / batch_size
            net.fc1.bias.grad = hidden_error.sum(dim=0) / batch_size


        optimizer.step()
        train_loss_hist.append(loss_val.item())

        # Test set 
        test_data_it, test_targets_it = next(iter(test_loader))
        test_data_it, test_targets_it = test_data_it.to(device).view(batch_size, -1), test_targets_it.to(device)

        test_outputs, _ = net(test_data_it)
        if test_outputs.dim() == 3:
            test_outputs = test_outputs.mean(dim=0)  #average over time

        test_log_probs = log_softmax_fn(test_outputs)
        test_loss = loss_fn(test_log_probs, test_targets_it).item()
        test_loss_hist.append(test_loss)

        # Accuracy 
        train_acc = (v_rec.argmax(dim=1) == targets_it).float().mean().item()
        test_acc = (test_outputs.argmax(dim=1) == test_targets_it).float().mean().item()
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)

    
        #if minibatch_counter % 20 == 0:
        if counter % 20 == 0:
            print(f"Epoch {epoch}, Minibatch {minibatch_counter}")
            print(f"Train Loss: {loss_val.item():.4f}, Test Loss: {test_loss:.4f}")
            print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}\n")

  
            np.savetxt(f'{directory}/train_loss_epoch{epoch}_{minibatch_counter}.txt', train_loss_hist)
            np.savetxt(f'{directory}/test_loss_epoch{epoch}_{minibatch_counter}.txt', test_loss_hist)
            np.savetxt(f'{directory}/train_acc_epoch{epoch}_{minibatch_counter}.txt', train_acc_hist)
            np.savetxt(f'{directory}/test_acc_epoch{epoch}_{minibatch_counter}.txt', test_acc_hist)

        minibatch_counter += 1
        counter += 1





