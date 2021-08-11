import numpy as np
import random

from model import QNetwork
from replaybuffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=0, 
                 gamma=0.99, alpha=5e-4, buffer_size=int(1e5), batch_size=64,
                 tau=1e-3, update_every=4, fc1_scalar=1, fc2_scalar=1):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            gamma (float): discount parameter
            alpha (float): learning rate
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            tau (float): interpolation parameter for soft update
            update_every (int): update every update_every step   
            fc1_scalar (float): size of hidden layer 1 = number of input features * fc1_scalar 
            fc2_scalar (float): size of hidden layer 2 = number of input features * fc2_scalar
        """

        self.state_size = state_size
        self.action_size = action_size

        # init seed
        random.seed(seed)
        np.random.seed(seed)

        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.update_every = update_every

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, fc1_scalar=fc1_scalar, fc2_scalar=fc2_scalar).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, fc1_scalar=fc1_scalar, fc2_scalar=fc2_scalar).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=alpha)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        # Break if it's not the time to learn or if there are not enough samples in memory. 
        if self.t_step != 0 or len(self.memory) < self.batch_size:
            return

        experiences = self.memory.sample()
        self.learn(experiences)
        

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_values = self.qnetwork_local.eval_forward(state)
        return self.epsilon_greedy(action_values, eps)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        next_action_values = self.qnetwork_target.forward(next_states)
        max_next_action_values = next_action_values.max(1)[0].unsqueeze(1)

        targets = rewards + self.gamma * max_next_action_values * (1 - dones)

        # predicted action values for given states and actions
        action_values = self.qnetwork_local.forward(states).gather(1, actions)

        # backpropagation
        loss = F.mse_loss(action_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def epsilon_greedy(self, values, eps=0.):
        """epsilon-greedy action selection

        Params
        ======
            values: action values
            eps: epsilon for greedy selection 
        """
        if random.random() > eps:
            return np.argmax(values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
