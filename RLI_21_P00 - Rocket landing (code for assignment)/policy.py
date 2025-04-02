import random
import numpy as np
import torch
import utils
import torch.optim as optim

import torch.nn as nn

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


class PositionalMapping(nn.Module):
    """
    Positional mapping Layer.
    This layer map continuous input coordinates into a higher dimensional space
    and enable the prediction to more easily approximate a higher frequency function.
    See NERF paper for more details (https://arxiv.org/pdf/2003.08934.pdf)
    NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (Aug/2020)
    """

    def __init__(self, input_dim, L=5, scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        self.scale = scale

    def forward(self, x):

        x = x * self.scale

        if self.L == 0:
            return x

        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x)
            x_cos = torch.cos(2**i * PI * x)
            h.append(x_sin)
            h.append(x_cos)

        return torch.cat(h, dim=-1) / self.scale


class MLP(nn.Module):
    """
    Multilayer perception with an embedded positional mapping (if L=0, then no positional mapping)
    """

    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_size=128, L=7):
        super().__init__()

        self.mapping = PositionalMapping(input_dim=input_dim, L=L)

        k = 1; self.add_module('linear'+str(k),nn.Linear(in_features=self.mapping.output_dim, out_features=hidden_size, bias=True)) # input layer
        for layer in range(hidden_layers):
            k += 1; self.add_module('linear'+str(k),nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True))
        k += 1; self.add_module('linear'+str(k),nn.Linear(in_features=hidden_size, out_features=output_dim, bias=True)) # output layer
        self.layers = [module for module in self.modules() if isinstance(module,nn.Linear)]

        negative_slope = 0.2; self.relu = nn.LeakyReLU(negative_slope)
        
        for child in self.named_children(): print(child)
        print(self.layers)
        

    def forward(self, x): # x: state
        # shape x: 1 x m_token x m_state
        x = x.view([1, -1])
        x = self.mapping(x)
        for k in range(len(self.layers)-1):
            x = self.relu(self.layers[k](x))
        # final output layer
        x = self.layers[-1](x)
        """
        in the case of the actor, the ouput layer has to be softmax-ed:
        x = policy_dist 
        x = F.softmax(self.layers[-1](x), dim=1)
        """
        return x

GAMMA = 0.99

class ActorCritic(nn.Module):
    """
    RL policy and update rules
    input_dim = num_inputs
    output_dim = num_actions

    Default configuration:
        hidden_layers=2
        hidden_size=128 
        positional mapping L=7
        learning_rate = 5e-5

    Other configurations to be tried (for simpler problems):
        hidden_layers=0
        hidden_size=256 
        No positional mapping L=0
        learning_rate = 3e-4
    """

    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_size=128, L=7, learning_rate=5e-5):
        super().__init__()

        self.output_dim = output_dim
        self.actor  = MLP(input_dim=input_dim, output_dim=output_dim, # output = num_actions
                          hidden_layers=hidden_layers, hidden_size=hidden_size, L=L)
        self.critic = MLP(input_dim=input_dim, output_dim=1,          # output = scalar value
                          hidden_layers=hidden_layers, hidden_size=hidden_size, L=L)
        self.softmax = nn.Softmax(dim=-1)

        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        #              = optim.Adam(self.parameters(),    lr=learning_rate)

    def forward(self, x):
        # shape x: batch_size x m_token x m_state
        y = self.actor(x)
        probs = self.softmax(y)
        value = self.critic(x)

        return probs, value

    def get_action(self, state, deterministic=False, exploration=0.01):

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs, value = self.forward(state)
        probs = probs[0, :]
        value = value[0]

        if deterministic:
            action_id = np.argmax(np.squeeze(probs.detach().cpu().numpy()))
        else:
            if random.random() < exploration:  # exploration
                action_id = random.randint(0, self.output_dim - 1)
            else:
                action_id = np.random.choice(self.output_dim, p=np.squeeze(probs.detach().cpu().numpy()))

        log_prob = torch.log(probs[action_id] + 1e-9)

        return action_id, log_prob, value

    @staticmethod
    def update_ac(network, rewards, log_probs, values, masks, Qval, gamma=GAMMA):

        # compute Q values
        Qvals = calculate_returns(Qval.detach(), rewards, masks, gamma=gamma)
        Qvals = torch.tensor(Qvals, dtype=torch.float32).to(device).detach()

        log_probs = torch.stack(log_probs)
        """
        ⚠️ unstitched code⚠️
        # In case of A2C to SAC:
        policy_dist = probs # returned by self.forward()
        dist = policy_dist.detach().numpy() 
        entropy = -np.sum(np.mean(dist) * np.log(dist))
        entropy_term += entropy
        """
        values = torch.stack(values)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss
        """
        ⚠️ unstitched code⚠️
        # In case of A2C to SAC:
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term
        """
        network.optimizer.zero_grad()
        ac_loss.backward()
        network.optimizer.step()

