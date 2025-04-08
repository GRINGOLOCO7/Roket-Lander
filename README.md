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


## 🔁 Actor-Critic Update Function: With vs. Without Entropy Regularization

This section compares two versions of the `update_ac` function used in Actor-Critic methods. The key difference lies in the inclusion of **entropy regularization** — a term used to encourage exploration in reinforcement learning.

---

### 📜 Code 1: With Entropy Regularization

<details>
<summary>CODE</summary>

```python
@staticmethod
def update_ac(network, rewards, log_probs, values, masks, Qval, probs_list, gamma=GAMMA):

    Qvals = calculate_returns(Qval.detach(), rewards, masks, gamma=gamma)
    Qvals = torch.tensor(Qvals, dtype=torch.float32).to(device).detach()

    log_probs = torch.stack(log_probs)
    values = torch.stack(values)
    advantage = Qvals - values

    # ✅ Entropy term computation
    entropies = []
    for probs in probs_list:
        entropy = -torch.sum(probs * torch.log(probs + 1e-9))
        entropies.append(entropy)
    entropy_term = torch.stack(entropies).mean()

    # ✅ Losses
    actor_loss = (-log_probs * advantage.detach()).mean()
    critic_loss = 0.5 * advantage.pow(2).mean()

    # ✅ Final loss includes entropy regularization
    ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

    network.optimizer.zero_grad()
    ac_loss.backward()
    network.optimizer.step()
```
</details>


### 📜 Code 2: Without Entropy Regularization
<details>
<summary>CODE</summary>

```python
@staticmethod
def update_ac(network, rewards, log_probs, values, masks, Qval, gamma=GAMMA):

    Qvals = calculate_returns(Qval.detach(), rewards, masks, gamma=gamma)
    Qvals = torch.tensor(Qvals, dtype=torch.float32).to(device).detach()

    log_probs = torch.stack(log_probs)
    values = torch.stack(values)
    advantage = Qvals - values

    # ❌ No entropy term
    actor_loss = (-log_probs * advantage.detach()).mean()
    critic_loss = 0.5 * advantage.pow(2).mean()

    # ❌ No entropy regularization added to loss
    ac_loss = actor_loss + critic_loss

    network.optimizer.zero_grad()
    ac_loss.backward()
    network.optimizer.step()
```
</details>

## 🔍 Side-by-Side Comparison

| Feature                         | Code 1: With Entropy        | Code 2: Without Entropy    |
|--------------------------------|-----------------------------|----------------------------|
| **Entropy Regularization**     | ✅ Included                  | ❌ Not included             |
| **Input: `probs_list`**        | ✅ Required (full distribution at each step) | ❌ Not required             |
| **Encourages Exploration**     | ✅ Yes (via entropy bonus)   | ❌ No (pure exploitation)   |
| **Loss Function**              | `actor_loss + critic_loss + 0.001 * entropy_term` | `actor_loss + critic_loss` |
| **Policy Behavior**            | Stochastic, exploratory      | Deterministic, greedy      |
| **Best Use Case**              | Complex tasks, sparse rewards | Simpler tasks, stable policies |

---

### 🧠 Why Use Entropy Regularization?

Entropy helps keep the policy **uncertain** and **diverse** during learning, which can prevent:
- Overconfidence in suboptimal actions
- Early convergence to poor strategies
- Lack of exploration in high-dimensional or sparse-reward environments

**Higher entropy** → More exploration
**Lower entropy** → More exploitation

> The entropy bonus encourages the agent to keep trying different actions by rewarding randomness in the policy.

---

### 📌 Summary

- ✅ Use **Code 1** because our environment benefits from **exploration**, especially in complex or deceptive reward landscapes.
- 🚫 Use **Code 2** for environments where exploration is less critical or when your agent already performs well with a more deterministic approach.

