import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from Actor import Actor, Critic
from noise import OUNoise
from model import CriticNetwork
from ReplayBuffer import ReplayBuffer


BUFFER_SIZE = int(1e6)  # replay buffer size
#BATCH_SIZE = 64        # minibatch size
GAMMA = 0.995            # discount factor
#TAU = 9e-3              # for soft update of target parameters
#LR_ACTOR = 1e-4         # learning rate of the actor
#LR_CRITIC = 5e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
#UPDATE_EVERY = 2

class Learner() :

    def __init__(self, state_size, action_size, random_seed, num_agents, device, hps):
        self.noise = OUNoise(action_size, random_seed)
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.count = 0
        # setting the hyperparameters
        self.batch_size = hps.batch_size
        self.tau = hps.tau
        self.lr_actor = hps.lr_actor
        self.lr_critic = hps.lr_critic
        self.update_every = hps.update_every
        # shared replay buffer
        self.memory = ReplayBuffer(BUFFER_SIZE,  self.batch_size, random_seed)

        # Critic networks - 1 network (local + target) per agent
        self.critics = [Critic(state_size, action_size, random_seed, self.lr_critic, WEIGHT_DECAY, device) for i in range(num_agents)]
        # Actor networks - 1 network (local + target) per agent
        self.actors = [Actor(state_size, action_size, random_seed, self.lr_actor, self.noise, device) for i in range(num_agents)]

    def act(self, all_states, noise_factor=1.0, add_noise=True) :
        actions = []
        for actor, state in zip(self.actors, all_states):
            action = actor.act(state, noise_factor=noise_factor, add_noise=add_noise)
            actions.append(action)
        return actions

    def reset(self) :
        self.noise.reset()

    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)

        self.count = (self.count + 1) % self.update_every
        if len(self.memory) > self.batch_size :
            if self.count == 0 :
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """ Updates the weights of actor and critic local networks using batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """    
        states, actions, rewards, next_states, dones = experiences # states is tensor 256 x 48
        states_agent = torch.split(states.view(-1, self.num_agents, self.state_size), 1, dim=1)    
        next_states_agent = torch.split(next_states.view(-1, self.num_agents, self.state_size), 1, dim=1)

        #-----------------------------  Update critic networks  --------------------------#
        # Get predicted next-state actions and Q values from target models
        next_actions_agent = [actor.actor_target(next_states_agent[i]) for i, actor in enumerate(self.actors)]
        next_actions = torch.cat(next_actions_agent, dim=1).view(-1, self.num_agents*self.action_size)
        rewards_agent = torch.split(rewards, 1, dim=1)
        dones_agent = torch.split(dones, 1, dim=1)

        for i in range(self.num_agents):
            Q_targets_next = self.critics[i].critic_target(next_states, next_actions) # all next states et all next actions !!!!!!
            Q_expected = self.critics[i].critic_local(states, actions) # all states and all actions !!!!!!
            Q_targets = rewards_agent[i] + (gamma * Q_targets_next * (1 - dones_agent[i]))
            # Compute critic loss
            critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
            # Minimize the loss
            self.critics[i].critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critics[i].critic_optimizer.step()

        #-----------------------------  Update actor networks (1 network per agent) --------------------------#
        for i in range(self.num_agents) :
            actions_pred_agent = [self.actors[j].actor_local(states_agent[j]) if j == i \
            else self.actors[j].actor_local(states_agent[j]).detach() for j in range(self.num_agents)]
            actions_pred = torch.cat(actions_pred_agent, dim=1).view(-1, self.num_agents*self.action_size)
            actor_loss = -self.critics[i].critic_local(states, actions_pred).mean()
            # Minimize the loss
            self.actors[i].actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actors[i].actor_optimizer.step()    

        #-----------------------------  Update target networks --------------------------#
        for i in range(self.num_agents):
            self.soft_update(self.actors[i].actor_local, self.actors[i].actor_target, self.tau)
            self.soft_update(self.critics[i].critic_local, self.critics[i].critic_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau) * target_param.data)  