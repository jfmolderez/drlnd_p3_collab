import numpy as np
import torch
import random
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """ Fixed size buffer to stor tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """ Initializes a ReplayBuffer object
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """        
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names = ['states', 'actions', 'rewards', 'next_states', 'dones'])
        self.seed = random.seed(seed)
        
    def add(self, states, actions, rewards, next_states, dones):
        """push into the buffer"""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from the buffer"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float().to(device)  
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)  
            
        return (states , actions, rewards, next_states, dones)
        

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)