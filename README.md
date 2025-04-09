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
8. `pip install stable-baselines3[extra] gym` for PPO and other algorithms

<br>


## Files


- `rocket_env.py`: Wrapper using `rocket.py` to make _Gymnasium_ compatible environment
- `test_env.py`: Script just to test that `rocket_env.py` wrapper works
- `policy.py`: Script for definign policy for SAC (using entropy term)
- `testbed_run.py`: Script to decide the task (_hover_ or _landing_) and the model checkpoint to use (_landing_ckpt_ or _landing_entropy_ckpt_)
- `testbed_train.py`: Script to train using `policy.py` (it will automaticly train with entropy term, and save the checpoints in _landing_entropy_ckpt_)
- `testbed_rewards.ipynb`: Plot model structure and rewards over training
- `/landing_ckpt`: Contains the checkpoints before adding entropy term to policy
- `/landing_entropy_ckpt`: Contains the checkpoints after adding entropy term to policy
- `PPO.py`: Uses the gymnasum compatible environment to trin starship how to land using PPO from stable baseline3

> [!CAUTION]
> `rocket_env_dummy.py`: Wrapper using `rocket.py` to make _Gymnasium_ compatible environment

---

![actor critic](https://github.com/GRINGOLOCO7/Roket-Lander/blob/main/RLI_21_P00%20-%20Rocket%20landing%20(code%20for%20assignment)/actor_critic.png)

<br>


# Policy Explenation

[policy.py](https://github.com/GRINGOLOCO7/Roket-Lander/blob/main/RLI_21_P00%20-%20Rocket%20landing%20(code%20for%20assignment)/policy.py)

## Actor-Critic Reinforcement Learning Implementation

This repository contains a PyTorch implementation of an Actor-Critic reinforcement learning model. The code is designed for training agents in environments like the `Rocket` simulation (hover or landing tasks).

### Overview
Actor-Critic is a reinforcement learning method that combines the strengths of both the policy-based and value-based methods. The Actor-Critic architecture combines two components:

- **Actor**: Decides which actions to take based on the current state by learning from a parametrized policy.
- **Value function (Critic)**: Evaluates how good those actions are by estimating a value function, which represents the expected reward of the given state.

In other words, the actor decides what action to carry out and the critic judges how good that decision was. The utilization of the Actor-Critic method enables the agent to optimize its policy by using the judgement and feedback of the Critic and increases learning efficiency, instead of using methods such as REINFORCE (trial-and-error).

### How does Actor–Critic combine Policy-based and Value-based methods?

| Component              | Role                                     | How does it learn?                           |
|------------------------|------------------------------------------|----------------------------------------------|
| Policy-based (Actor)   | Learns the action selection policy π(a\|s) | Optimized using gradients from the critic    |
| Value-based (Critic)   | Learns the value function of each state  | Trained using temporal difference (TD) errors |

By applying the combination of these to methods, Actor-Critic can take advantage of the strengths of both, and this will result in a model that has:
- Low degree of variance due to bootstrapping perform by the critic
- Online learning use either 1-step or multi-step returns
- A good balance or tradeoff between bias and variance through temporal difference learning



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

## Deep Dive into `update_ac`

Before implementing the Entropy Regularizatoin, we have to better understand how the `update_ac` code works.

The `update_ac()` function represents the main function that contributes towards the learning process in the Actor-Critic implementation in the code. It updates the parameters of both the actor and the critic as the model learns and finds the best actions with high value, while using a combination of gradients derived from the returns and the estimated value, and this can be seen in the following code snippet:

The function also uses the concept of the advantage. The advantage basically quantifies how much better or worse an action that is being performed is in comparison with the expected value of the state.
```python
advantage = Qvals - values
```
This uses the discounted return Q as a Monte Carlo estimate of Q(s,a) and the value function is predicted by the critic. This line tells the actor whether an action carried out is better or worse than expected and by how much, acting as a guide towards how to change the direction and magnitude when updating the policy.

### Actor's Loss
```python
actor_loss = (-log_probs * advantage.detach()).mean()
```

This is derived from the policy gradient theorem, which states that a policy parameter can be updated using the equation:

$$\nabla J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi(a \mid s; \theta) \cdot A(s, a) \right]$$

The log_probs is the log part of the equation, and the advantage is represented as the scalar signal that reflects how good or bad the action was. When the advantage is used, it is associated with the .detach() function to ensure that it is treated as a constant value during backpropagation when running the policy, which prevents gradients from the actor’s loss to influence the critic’s decision.

To calculate the actor loss, we multiplied the two terms and took the negative mean. The negative mean is required because Pytorch uses this value to minimize the loss function and maximize the policy performance.

### Critic's Loss
```python
critic_loss = 0.5 * advantage.pow(2).mean()
```

The critic’s loss is basically the mean squared error between the predicted value and the target value. This is because the critic estimates the state’s value and wants the theoretical value to be as close to the actual return, Q(s,a), and in return minimize the MSE between the predicted/theoretical value and the actual value.

We then compute the total loss
```python
ac_loss = actor_loss + critic_loss
```
To perform backpropagation by first resetting the system from all previous gradients, computing the gradient loss and then updating the network weights the network.optimizer.step(), to give weights to the actions that maximize the policy performance. This in turn will update both the actor and critic network parameters simultaneously using the combination of both the actor and critic losses. This is done by:
1. network.optimizer.zero_grad() → removes the gradient from the previous run
2. ac_loss.backward() → computes the new gradients using backpropagation with respect to the parameters of both the actor and critic
3. network.optimizer.step() → updates the model parameters based on the gradients calculated in the previous step to help improve the policy and value function.



## 🔁 Actor-Critic Update Function: With vs. Without Entropy Regularization

This section compares two versions of the `` function used in Actor-Critic methods. The key difference lies in the inclusion of **entropy regularization** — a term used to encourage exploration in reinforcement learning.

---

### 📜 Code 1: With Entropy Regularization

<details>
<summary>CODE</summary>

```python
@staticmethod
def update_ac(network, rewards, log_probs, values, masks, Qval, probs_list, gamma=GAMMA):

    Qvals = calculate_returns(Qval.detach(update_ac), rewards, masks, gamma=gamma)
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

## Entropy Regularization Explanation
This is a regularization technique used in reinforcement learning to encourage the policy to stay random rather than becoming too specific and eventually overfitting and becoming deterministic. This concept is taken from thermodynamics, where entropy is a measure of the degree of randomness of the particles in a molecule. Entropy regularization is similar, since it depends on how many random actions or the randomness of the policy. High entropy means high randomness, so the model has explored multiple actions (exploration), and low entropy means lower randomness, so the model explored only a few actions (exploitation). Therefore, entropy regularization helps balance the trade-off between exploration and exploitation.

Through using entropy regularization, the model can explore more diverse actions and have better exploration earlier on in training, and avoid being stuck at the local optima or continuously picking the same actions.

The entropy of the policy is calculated using the equation:

$$- \sum_a \pi(a \mid s) \log \pi(a \mid s)$$

Which was implemented in our code in the form of
```python
entropy = -torch.sum(probs * torch.log(probs + 1e-9))
```

The entropy is then subtracted from the total loss:
```python
ac_loss = actor_loss + critic_loss + 0.001 * entropy_term  # entropy regularization
```

And this follows the equation

$$\text{Total Loss} = \text{Actor Loss} + \text{Critic Loss} - \beta \cdot \mathcal{H}(\pi)$$

Where beta, or 0.001 in the case of our code, encourages exploration and looking into multiple new actions.

In the context of the Actor-Critic, this is very crucial because without entropy regularization the actor might converge and choose one action without exploring all possible options and this can lead to suboptimal policies, especially in complex environments. But after adding entropy regularization to the code, the actor is encouraged to explore all possible actions during the entire process of training and we can find better solutions due to further explorations. Also, this is crucial because the critic adds stability to the agent, while entropy ensures that the actor continues being stochastic and continues exploring.



## 🔍 Side-by-Side Comparison

| Feature                         | Code 1: With Entropy        | Code 2: Without Entropy    |
|--------------------------------|-----------------------------|----------------------------|
| **Entropy Regularization**     | ✅ Included                  | ❌ Not included             |
| **Input: `probs_list`**        | ✅ Required (full distribution at each step) | ❌ Not required             |
| **Encourages Exploration**     | ✅ Yes (via entropy bonus)   | ❌ No (pure exploitation)   |
| **Loss Function**              | `actor_loss + critic_loss + 0.001 * entropy_term` | `actor_loss + critic_loss` |
| **Policy Behavior**            | Stochastic, exploratory      | Deterministic, greedy      |
| **Best Use Case**              | Complex tasks, sparse rewards | Simpler tasks, stable policies |


<p align="center">
  <div style="display: flex; flex-direction: row; justify-content: center; gap: 20px;">
    <div style="text-align: center;">
      <p><strong>avg reward with entropy</strong></p>
      <img src="https://github.com/GRINGOLOCO7/Roket-Lander/raw/main/RLI_21_P00%20-%20Rocket%20landing%20(code%20for%20assignment)/avg_reward_with_entropy.png" width="400"/>
    </div>
    <div style="text-align: center;">
      <p><strong>avg reward without entropy</strong></p>
      <img src="https://github.com/GRINGOLOCO7/Roket-Lander/raw/main/RLI_21_P00%20-%20Rocket%20landing%20(code%20for%20assignment)/avg_reward_without_entropy.png" width="400"/>
    </div>
  </div>
</p>



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


# Converting Environment to OpenAI Gymnaisum Compatible Version.
There are multiple approaches that we can take to convert the rocket_env.py to one that implements an Open-AI gymnasium compatible version of rocket.py environment. The three approaches we were told to investigate conversions such as refactoring, wrapping and subclassing. Before investigating the pros and cons of each conversion method, we explored the pros and cons of using a gymnasium compatible environment.

## Pros of using Gymnasium

1. Introduces modularity to the code and makes it easier to plug the environment into training loops, wrappers and evaluators.

2. Using gymnasium allows the environment to be reused for multiple frameworks, meaning that once we developed the changes for gymnasium in the rocket_env.py, we can use it for multiple algorithms and frameworks, so it can be reused for PPO, A2C, SAC, etc. 

3. Can be implemented on multi-agent reinforcement learning agents.

4. Works efficiently and effectively with existing training pipelines, such as stable-baseline3 which we are going to be using in the next part.

5. Allows clear differentiation between the simulation logic and the RL wrappers, which enhances the modularity of the code.

6. Promotes consistency and interoperability across projects or files within the same project.

7. Allows us to build models such as SAC, A2C, PPO in an out-of-the-box fashion and build functions for such models accordingly.

## Cons of using Gymnasium
1. Requires restructuring chunks of the original code to match the gymnasium API

2. Must adhere to the structure and API of gymnasium so that the environment runs correctly with no errors

3. Debugging can be difficult because one error in the environment setup can propagate throughout the entire wrapper

4. The render modes of the environment must be implemented correctly because any missing rendering logic will affect the integration of the environment with the RL logic and the model’s evaluation. 

After researching and discussing the pros and cons of gymnasium, we can see that converting an environment to something that is Open-AI gymnasium compatible has a lot of pros and can help us build functions for algorithms (PPO, SAC, A2C, etc...) out of the box and facilitates the process of building these models.

The second part was to find the best type of conversion so that we just replace the initial code we have with an Open-AI gymnasium compatible version, and we were given three recommendations to choose from, refactoring, wrapping and subclassing. Firstly, we compared these conversions to see which one is best

## Which Type of Conversion to Use?
### Conversion Approaches

| Type of Conversion | Description | Pros | Cons |
|--------------------|-------------|------|------|
| **Refactoring** | TThis involves merging the rocket logic into the environment itself | It simplifies the process of calling on the methods | It does not support modularity in the code and will be harder to maintain and resolve errors because of it being implemented directly in the code for the RL agent |
| **Wrapping** | This involves creating an adapter class around the `rocket.py` file. | It simplifies the process by decoupling the logic from the environment. | Increases the presence of boilerplates, meaning that sections of the code will be repeated multiple times, as they are needed for the functionality but not for the agent logic. |
| **Subclassing** | This involves directly subclassing the Gym environment and using a `Rocket` object to delegate the core environment logic, allowing us to reuse the methods in Rocket by just calling them. |  It is simple to implement because it maintains the logic and functionality of `rocket.py` and allows a clear separation between the logic and functionality as well. |  It requires shaping some of the `rocket.py` API so that it is 100% compatible with all files of the RL agent. |


We decided to use **subclassing** as it gave us full control over the Gym API while keeping the original rocket.py simulation intact and reusable for all possible functionalities of the code. Also, thai ensures the code is modular and is fully compatible with the Gym API.
