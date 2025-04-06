# Roket-Lander
RL for teaching roket to land using **Soft Actor Critic** (SAC)

----

# Set Up Environment

On Window:

1. Conda environment with `Gymnasium`
2. Install `torch` (`pip install torch torchvision torchaudio`)
3. Install `openCV` (`pip install opencv-python`)
4. Install `matplotlib`
5. If needed downgrade `numpy` to be compatible with functions used (`pip install numpy==1.26.4`)
6. Install `graphviz` and add to EnvVariables
7. Install andy other libraries required (just use `pip`)

<br>



---



<br>


# Policy Explenation

[policy.py](https://github.com/GRINGOLOCO7/Roket-Lander/blob/main/RLI_21_P00%20-%20Rocket%20landing%20(code%20for%20assignment)/policy.py)

## Actor-Critic Reinforcement Learning Implementation

This repository contains a PyTorch implementation of an Actor-Critic reinforcement learning model. The code is designed for training agents in environments like the `Rocket` simulation (hover or landing tasks).

### Overview

The Actor-Critic architecture combines two components:

- **Actor**: Decides which actions to take
- **Value function (Critic)**: Evaluates how good those actions are

This implementation features:

- Positional encoding for better representation of continuous state spaces
- MLP (Multi-Layer Perceptron) networks for both actor and critic
- Configurable hyperparameters for various environments

### Core Components

<details>
<summary>Actor-Critic Model Architecture</summary>

The `ActorCritic` class combines policy (actor) and value (critic) networks:

```python
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_size=128, L=7, learning_rate=5e-5):
        # input_dim: dimension of state space
        # output_dim: dimension of action space
        # L: positional encoding parameter

        self.actor = MLP(input_dim, output_dim)  # Decides actions
        self.critic = MLP(input_dim, 1)          # Estimates value
```

The actor outputs action probabilities, while the critic estimates the value of being in a particular state.

</details>

<details>
<summary>Positional Encoding</summary>

The `PositionalMapping` class implements a technique borrowed from NeRF (Neural Radiance Fields) that helps networks better represent high-frequency functions:

```python
class PositionalMapping(nn.Module):
    def __init__(self, input_dim, L=5, scale=1.0):
        # L: number of frequency bands

    def forward(self, x):
        # Maps input to higher dimension using sin/cos functions
        # at different frequencies
```

This encoding is particularly useful for continuous control tasks, as it helps the network learn more complex behaviors.

</details>
<details>
<summary>MLP (Multi-Layer Perceptron)</summary>

The `MLP` class builds neural networks with configurable depth and width:

```python
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_size=128, L=7):
        # Apply positional mapping first
        self.mapping = PositionalMapping(input_dim, L)

        # Build network layers
        # ...
```

Each MLP includes:

- Positional encoding of inputs
- Configurable number of hidden layers
- LeakyReLU activation functions

![actor_critic](https://github.com/GRINGOLOCO7/Roket-Lander/blob/main/RLI_21_P00%20-%20Rocket%20landing%20(code%20for%20assignment)/actor_critic.png)

</details>
<details>
<summary>RL Training Process</summary>

The `update_ac` method implements the Actor-Critic update algorithm:

```python
@staticmethod
def update_ac(network, rewards, log_probs, values, masks, Qval, gamma=GAMMA):
    # Calculate returns (future discounted rewards)
    Qvals = calculate_returns(Qval.detach(), rewards, masks, gamma=gamma)

    # Calculate advantage (how much better/worse an action was than expected)
    advantage = Qvals - values

    # Update both networks
    actor_loss = (-log_probs * advantage.detach()).mean()
    critic_loss = 0.5 * advantage.pow(2).mean()
    ac_loss = actor_loss + critic_loss
```

Key concepts:

- Returns: Sum of discounted future rewards
- Advantage: Difference between actual returns and predicted value
- Actor loss: Encourages actions that led to better-than-expected outcomes
- Critic loss: Reduces prediction error of the value function

### Hyperparameters
The default configuration works well for complex control tasks:

- Hidden layers: 2
- Hidden size: 128
- Positional encoding: L=7
- Learning rate: 5e-5
- Discount factor (gamma): 0.99

For simpler problems, consider:

- Hidden layers: 0
- Hidden size: 256
- No positional mapping (L=0)
- Learning rate: 3e-4