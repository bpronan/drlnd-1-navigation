import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """ Q-Network model. """

    def __init__(self, state_size, action_size, seed, hidden_sizes=[64, 64]):
        """ Initialize parameters and define PyTorch model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_sizes (list): Output dimensions of the hidden layers
        """
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DuelingQNetwork(nn.Module):
    """ Dueling DQN model. """

    def __init__(self, state_size, action_size, seed, hidden_sizes=[64, 64]):
        """ Initialize parameters and define PyTorch model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_sizes (list): Output dimensions of the hidden layers
        """
        super(DuelingQNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, hidden_sizes[0])

        # advantage network
        self.adv_1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.adv_2 = nn.Linear(hidden_sizes[1], action_size)

        # value network
        self.val_1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.val_2 = nn.Linear(hidden_sizes[1], 1)


    def forward(self, x):
        x = F.relu(self.fc1(x))

        adv = F.relu(self.adv_1(x))
        adv = self.adv_2(adv)

        val = F.relu(self.val_1(x))
        val = self.val_2(val)

        return val + adv - adv.mean()
