# Project 1: Navigation Report
by Dhruv Baldawa

## Overview

A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue
 banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

`0` - move forward.
`1` - move backward.
`2` - turn left.
`3` - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100
 consecutive episodes.

## Learning Algorithm

The algorithm implemented for learning is a Deep Q Learning algorithm which uses:
 - Deep neural network for learning
 - Epsilon-greedy method for policy selection
 - Fixed Q-targets
 - Experience replays

It is a policy-based estimation model which tries to evaluate the optimal action policy to maximize the discounted
, cumulative reward.

### Methods applied

#### Deep Neural Networks

Our model uses a deep neural network with the following layers:

* Input layer (number of states = 37 nodes)
* ReLU Layer
* Fully-connected layer (64 nodes)
* ReLU Layer
* Fully-connected layer (32 nodes)
* ReLU Layer
* Fully-connected layer (number of actions = 4 nodes)

#### Epsilon-greedy policy selection

We will select an action accordingly to an epsilon greedy policy. Simply put, we’ll sometimes use our model for
 choosing the action, and sometimes we’ll just sample one uniformly. The probability of choosing a random action will
  start at `policy.start` and will decay exponentially towards `policy.end`. `policy.decay` controls the rate of the
   decay.

```
policy = GreedyPolicy(
    start=1.0, 
    end=0.01, 
    decay=0.995,
)
```

#### Fixed Q-targets

We use a target neural network similar to our current neural network and update the parameters in the target neural
 network instead of directly updating the policy neural network. This is to minimize harmful correlations. We use a
  hyperparameter `TAU` to control the parameters we soft update in the target neural network.

#### Experience Replay

We’ll be using experience replay memory for training our DQN. It stores the transitions that the agent observes, allowing us to reuse this data later. By sampling from it randomly, the transitions that build up a batch are decorrelated. It has been shown that this greatly stabilizes and improves the DQN training.

### Hyperparameters
```
BUFFER_SIZE = int(1e5)    # replay buffer size
BATCH_SIZE = 32           # minibatch size
GAMMA = 0.99              # discount factor
TAU = 1e-3                # for soft update of target parameters
LR = 5e-4                 # learning rate
UPDATE_EVERY = 4          # how often to update the network
HIDDEN_LAYERS = (64, 32)  # hidden layers for the network
```

### Performance

The agent was able to reach an average score of +13 over 100 consecutive episodes in **< 400** episodes.

## Future work
* **Double DQN** - Deep Q-Learning tends to overestimate action values. Double Q-Learning has been shown to work well in practice to help with this.
* **Prioritized Experience Replay** - Deep Q-Learning samples experience transitions uniformly from a replay memory. Prioritized experienced replay is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability.
* **Dueling DQN** - Currently, in order to determine which states are (or are not) valuable, we have to estimate the
 corresponding action values for each action. However, by replacing the traditional Deep Q-Network (DQN) architecture with a dueling architecture, we can assess the value of each state, without having to learn the effect of each action.

