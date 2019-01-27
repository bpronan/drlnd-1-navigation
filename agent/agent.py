import numpy as np
import random

from .model import QNetwork, DuelingQNetwork
from .utils import ReplayBuffer

import torch
import torch.optim as optim

BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, state_size, action_size, double_dqn=False, dueling=False, seed=0):
        """ Initializes the DQN Agent.

        Params
        ======
            state_size (int): dimensions in the state space
            action_size (int): action space dimension
            double_dqn (bool): use the double DQN algorithm for calculating the q values
            dueling (bool): use the dueling DQN network architecture
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # use the dueling network or the sequential network
        if dueling:
            self.qnetwork_online = DuelingQNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        else:
            self.qnetwork_online = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_online.parameters(), lr=LR)
        self.criterion = torch.nn.MSELoss()

        self.replay = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, device, seed)
        self.update_t = 0

        self.double_dqn = double_dqn


    def step(self, state, action, reward, next_state, done):
        """ Learning step for the agent """

        # save the step in the replay buffer
        self.replay.add(state, action, reward, next_state, done)

        # learn update_every time steps
        self.update_t += 1
        self.update_t = self.update_t % UPDATE_EVERY
        if self.update_t == 0:
            if len(self.replay) >= BATCH_SIZE:
                experiences = self.replay.sample()
                self.learn(experiences, GAMMA)


    def act(self, state, eps=0.):
        """ Returns actions for the state based on the current policy

        Params
        ======
            state (array_like): current state of the environment
            eps (float): epsilon for epsilon-greedy action selection
        """
        # move the state to the device
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # get the action values from the online network
        self.qnetwork_online.eval()
        with torch.no_grad():
            action_values = self.qnetwork_online(state)
        self.qnetwork_online.train()

        # return the state based on the epsilon greedy policy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        if self.double_dqn:
            # get the max action index from the online network and the q-value from the target network
            next_actions = self.qnetwork_online(next_states).detach().argmax(1).unsqueeze(1)
            q_next = self.qnetwork_target(next_states).gather(1, next_actions)
        else:
            q_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        q_target = rewards + (gamma * q_next * (1 - dones))
        q_expected = self.qnetwork_online(states).gather(1, actions)

        # compute and minimize the loss
        loss = self.criterion(q_expected, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update the target network
        self.soft_update(self.qnetwork_online, self.qnetwork_target, TAU)

    def soft_update(self, online_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_online + (1 - τ)*θ_target

        Params
        ======
            online_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
