from unityagents import UnityEnvironment
import numpy as np
import torch
from Learner import Learner
from collections import deque, namedtuple
import itertools


def train(agent, n_episodes=2000, print_every=100):

    scores_deque = deque(maxlen=100)
    scores = []
    
    # noise management
    noise_factor = 1.0
    noise_reduction = 0.9999
    t_stop_noise = 12000
    
    # index of step across episodes
    i_step = 0

    # results
    i_episode_max = 0
    max_avg_score = 0.
    
    # main loop
    for i_episode in range(1, n_episodes + 1) : 

        collected_rewards = []
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        all_states = env_info.vector_observations             # get the current state (for each agent)
        agent.reset()                                         # resets the noise
        
        # running an episode until end
        while(True) : 
            if i_step < t_stop_noise :
                all_actions = agent.act(all_states, noise_factor=noise_factor)
                #print("training - all_states = ", all_states)
                #print("training - all_actions = ", all_actions)
                noise_factor *= noise_reduction
            else :
                all_actions= agent.act(all_states, add_noise=False)
            
            env_info = env.step(all_actions)[brain_name]
            all_next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                             # get reward (for each agent)
            dones = env_info.local_done                            # see if episode finished
            
            # store experiences & learn
            states = np.array(all_states).reshape(1,-1)
            actions = np.array(all_actions).reshape(1,-1)
            next_states = np.array(all_next_states).reshape(1,-1)
            agent.step(states, actions, rewards, next_states, dones) 
            
            all_states = all_next_states
            collected_rewards.append(rewards)
            i_step += 1
            
            if np.any(dones):
                break

        rewards = np.transpose(np.array(collected_rewards)) # for 1 episode
        scores_agent = np.sum(rewards, axis = 1)
        scores.append(np.max(scores_agent))
        scores_deque.append(np.max(scores_agent))
        avg_score = np.mean(scores_deque)
        if avg_score > max_avg_score :
            max_avg_score = avg_score
            i_episode_max = i_episode
        #print('\rEpisode {} \t Average Score : {:.5f}'.format(i_episode, avg_score, end=""))
        if  i_episode % print_every == 0 :
            print('\rEpisode {} \t Average Score : {:.5f}'.format(i_episode, avg_score))

        if max_avg_score > 1.5 :
            break

    return max_avg_score, i_episode_max


if __name__ == '__main__' :
    env = UnityEnvironment(file_name="Tennis", no_graphics=True)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents 
    num_agents = len(env_info.agents)
    #print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    #print('Size of each action:', action_size)

    # examine the state space 
    all_states = env_info.vector_observations
    state_size = all_states.shape[1]
    #print('There are {} agents. Each observes a state with length: {}'.format(all_states.shape[0], state_size))
    
    # train(learner, n_episodes=10000)

    batch_sizes = [32]#[16, 32, 64]
    taus = [9e-3]   #[5e-3, 9e-3, 1e-2, 2e-2]
    lr_actors = [5e-5] #[5e-5, 1e-4, 5e-4, 1e-3]
    lr_critics = [1e-4] #[1e-4, 5e-4]
    update_everys = [2]

    configs = []
    params = [batch_sizes, taus, lr_actors, lr_critics, update_everys]
    Hyperparams = namedtuple('Hyperparams',['batch_size', 'tau', 'lr_actor', 'lr_critic', 'update_every'])
    for params in itertools.product(*params) :
        configs.append(Hyperparams(*params))

    random_seed = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_max = 0.0
    fout = open("results.txt", "w")
    for config in configs :
        learner = Learner(state_size, action_size, random_seed, num_agents, device, config)
        max_score, episode = train(learner, n_episodes=10000)
        if max_score > best_max :
            best_max = max_score
            best_config = config
        print("******************************************")
        fout.write("******************************************")
        print("Hyperparameters : {}".format(config))
        fout.write("Hyperparameters : {}\n".format(config))
        print("Reached a maximum average of scores (over last 100 episodes) = {:.2f} after {:d} episodes".format(max_score, episode))
        fout.write("Reached a maximum average of scores (over last 100 episodes) = {:.2f} after {:d} episodes\n".format(max_score, episode))
        print("\t\t --------------- \n\n")
        fout.write("\t\t --------------- \n\n")
    print("Maximal score = ", best_max)
    fout.write("Maximal score = {:.2f}\n".format(best_max))
    print("Best configuration = ", best_config)
    fout.write("Best configuration = {}\n".format(best_config))
    fout.close()








