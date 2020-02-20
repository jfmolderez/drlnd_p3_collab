import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F

def hidden_init(layer):
    """ Initialisation of hidden layers"""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorNetwork(nn.Module):
    """
    The Actor network is an approximation of the policy function that maps states to actions
    """ 
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, normalize=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()

        self.normalize =normalize
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)   
        
    def forward(self, state):
        if state.dim() == 1 :
            state = torch.unsqueeze(state, 0)

        if not self.normalize :
            x = F.leaky_relu(self.fc1(state))
            x = F.leaky_relu(self.fc2(x)) 
        else :  
            x = self.fc1(state)
            if x.dim() == 3 :
                x = torch.squeeze(x)
            x = F.leaky_relu(self.bn1(x))
            x = F.leaky_relu(self.bn2(self.fc2(x)))
        
        return F.tanh(self.fc3(x))

class CriticNetwork(nn.Module):
    """
    The Critic network is an approximation of the action-value function 
    (Q function) that maps state-action pairs to value
    Since this is a collaborative context, the critic receives as inputs the states and actions
    for both agents
    """ 
    def __init__(self, state_size, action_size, seed, num_agents=2, fc1_units=256, fc2_units=256, normalize=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        input_size = num_agents * (state_size + action_size)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1) 
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()

        self.normalize = normalize
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
         
    def forward(self, states, actions):
        xs = torch.cat((states, actions), dim=1)
        if not self.normalize :
            x = F.leaky_relu(self.fc1(xs))
            x = F.leaky_relu(self.fc2(x))
        else :
            x = F.leaky_relu(self.bn1(self.fc1(xs)))
            x = F.leaky_relu(self.bn2(self.fc2(x)))
        return self.fc3(x)
