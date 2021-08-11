import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_scalar=4, fc2_scalar=8):

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1_size = int(state_size * fc1_scalar)
        self.fc2_size = int(state_size * fc2_scalar)
        
        self.fc1 = nn.Linear(state_size, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.fc3 = nn.Linear(self.fc2_size, action_size)

        print(f' features: [Input: {state_size}, Layer 1: {self.fc1_size}, Layer 2: {self.fc2_size}, output: {action_size} ]')
        

    def forward(self, state):
       
        x = self.fc1(state)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        return x

    
    def eval_forward(self, state):
        
        self.eval()
        with torch.no_grad():
            values = self.forward(state)
        self.train()
        return values