# Project 2: Continuous Control

by Dhruv Baldawa

## Overview

For this project, we will work with the Reacher environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The project submission need only solve one of the two versions of the environment.

### Option 1: Solve the First Version
The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

### Option 2: Solve the Second Version
The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
This yields an average score for each episode (where the average is over all 20 agents).
As an example, consider the plot below, where we have plotted the average score (over all 20 agents) obtained with each episode.

**The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.**

## Learning Algorithm

### Methods applied

#### Distributed Deep Deterministic Policy Gradient (D3PG)

We used the D3PG, or Distributed DDPG strategy to solve the **Option 2** of this problem. We added distributional updates to the DDPG algorithm, combined with the use of multiple distributed workers all writing into the same replay table.

The Deep Deterministic Policy Gradient (DDPG) strategy is used in this solution. It uses 2 deep neural networks - one of the actor and the other one for the critic with the following configuration:

##### Actor network configuration
* Input layer (number of states = 33 nodes) - batch normalized
* Leaky ReLU Layer
* Fully-connected layer (512 nodes)
* Leaky ReLU Layer
* Fully-connected layer (512 nodes)
* Output Layer (number of actions = 4 nodes)

##### Critic network configuration
* Input layer (number of states = 33 nodes) - batch normalized
* ReLU activation layer
* Fully-connected layer (516 (512 + number of actions) nodes)
* ReLU activation layer
* Fully-connected layer (512 nodes)
* Output Layer (1 node)

The agent learns every 20 timesteps and then we train the agent with 10 random samples.

#### Experience Replay

We’ll be using experience replay memory for training our network. It stores the transitions that the agent observes, allowing us to reuse this data later. By sampling from it randomly, the transitions that build up a batch are decorrelated. It has been shown that this greatly stabilizes and improves the training.

The experiences from all the agents is collected in a single buffer, and samples from this buffer are batched randomly and used to train the networks.

#### Ornstein-Uhlenbeck Process

We added noise in the network to make it explore more. We are decaying this noise gradually to reduce the effect of noise as the agent is being trained, this also helps the agent to learn faster.

#### Gradient Clipping

Large updates to weights during training can cause a numerical overflow or underflow. To reduce this instability while training, we rescale the gradient function to reduce unwanted effects on the training.

### Hyperparameters

```python
BUFFER_SIZE = int(1e6)           # replay buffer size
BATCH_SIZE = 128                 # minibatch size
GAMMA = 0.99                     # discount factor
TAU = 1e-3                       # for soft update of target parameters
LR_ACTOR = 1e-4                  # learning rate of the actor
LR_CRITIC = 1e-3                 # learning rate of the critic
WEIGHT_DECAY = 0                 # L2 weight decay
UPDATE_EVERY = 20                # number of timesteps before updating the network
UPDATE_NETWORK = 10              # how many times to update the network
EPSILON = 1                      # epsilon
EPSILON_DECAY = 1e-6             # rate at which we should decay epsilon
MAX_CLIPPING_NORMALIZATION = 1.  # gradient clipping normalization max value

CUTOFF_SCORE = 30.0
CUTOFF_WINDOW = 100

NOISE_MU = 0.
NOISE_THETA = 0.2
NOISE_SIGMA = 0.2
```

## Result

We were able to solve the **Option 2** of the environment and reach an average of +30 over 100 episodes (over 20 agents) in **219 episodes**

![rewards.png]

![agents-in-action.gif]

## Future Work
- As mentioned in [Gabriel Barth-Maron, Matthew W. Hoffman, 2018](https://openreview.net/pdf?id=SyZipzbCb) using D4PG and combining the current implementation with a Prioritized Experience Replay buffer can help increase the performance of the model.

## References
- [Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra; Continuous control with deep reinforcement learning, 2015](https://arxiv.org/abs/1509.02971)
- [Gabriel Barth-Maron˚, Matthew W. Hoffman˚, David Budden, Will Dabney,
Dan Horgan, Dhruva TB, Alistair Muldal, Nicolas Heess, Timothy Lillicrap; Distributed Distributional Deterministic Policy Gradients, 2018](https://openreview.net/pdf?id=SyZipzbCb)
- [Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel; Benchmarking Deep Reinforcement Learning for Continuous Control, 2015](https://arxiv.org/abs/1604.06778)
