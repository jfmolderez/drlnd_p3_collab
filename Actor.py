import random
import numpy as np 
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import ActorNetwork, CriticNetwork

def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Actor() :

    def __init__(self, state_size, action_size, random_seed, learning_rate, noise, device):
        self.state_size = state_size
        self.action_size = action_size
        self.seed= random.seed(random_seed)
        self.learning_rate = learning_rate

        self.actor_local = ActorNetwork(state_size, action_size, random_seed).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, random_seed).to(device)
        hard_update(self.actor_target, self.actor_local)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate)

        self.noise = noise
        self.device = device

    def act(self, state, noise_factor, add_noise):
        """ Returns actions for given state as per given policy"""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad() :
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise :
            action += (noise_factor * self.noise.sample())
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

class Critic() :

    def __init__(self, state_size, action_size, random_seed, learning_rate, weight_decay, device):

        self.state_size = state_size
        self.action_size = action_size
        self.seed= random.seed(random_seed)
        self.learning_rate = learning_rate

        self.critic_local = CriticNetwork(state_size, action_size, random_seed).to(device)
        self.critic_target = CriticNetwork(state_size, action_size, random_seed).to(device)
        hard_update(self.critic_target, self.critic_local)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate, weight_decay=weight_decay)



