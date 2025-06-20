import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import random

#Ensure results folder
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#Environment
class MiniGridEnv:
    def __init__(self):
        self.state_space = 5  # 5 states
        self.action_space = 2  # 0: left, 1: right
        self.reset()

    def reset(self):
        self.state = 2  # start in the middle
        return self.state

    def step(self, action):
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.state_space - 1, self.state + 1)

        done = self.state in [0, self.state_space - 1]
        reward = 1 if self.state == self.state_space - 1 else 0
        return self.state, reward, done

#Q-Network with Dropout
class DropoutQNet(nn.Module):
    def __init__(self, n_states, n_actions, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(n_states, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

#One-hot state representation
def one_hot(state, size):
    vec = torch.zeros(size)
    vec[state] = 1.0
    return vec

#Epsilon-Greedy Agent 
class EpsilonGreedyAgent:
    def __init__(self, env, qnet, eps=0.1, gamma=0.99):
        self.env = env
        self.qnet = qnet
        self.eps = eps
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=0.01)

    def select_action(self, state):
        if random.random() < self.eps:
            return random.choice([0, 1])
        with torch.no_grad():
            q_vals = self.qnet(one_hot(state, self.env.state_space).to(DEVICE))
            return torch.argmax(q_vals).item()

    def train(self, episodes=500):
        rewards = []
        for _ in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                total_reward += reward

                # Q-learning update
                q_vals = self.qnet(one_hot(state, self.env.state_space).to(DEVICE))
                next_q = self.qnet(one_hot(next_state, self.env.state_space).to(DEVICE)).detach()
                target = reward + self.gamma * torch.max(next_q)
                loss = F.mse_loss(q_vals[action], target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                state = next_state
            rewards.append(total_reward)
        return rewards

#Thompson Sampling Agent
class DropoutThompsonAgent(EpsilonGreedyAgent):
    def select_action(self, state):
        self.qnet.train()  # Keep dropout ON
        q_vals = self.qnet(one_hot(state, self.env.state_space).to(DEVICE))
        return torch.argmax(q_vals).item()

#Run & Compare
def run_experiment():
    env = MiniGridEnv()

    eps_agent = EpsilonGreedyAgent(env, DropoutQNet(5, 2).to(DEVICE))
    ts_agent = DropoutThompsonAgent(env, DropoutQNet(5, 2).to(DEVICE))

    print("Training epsilon-greedy agent...")
    eps_rewards = eps_agent.train()

    print("Training dropout + Thompson agent...")
    ts_rewards = ts_agent.train()

    plt.figure(figsize=(10, 5))
    plt.plot(np.convolve(eps_rewards, np.ones(10)/10, mode='valid'), label="Epsilon-Greedy")
    plt.plot(np.convolve(ts_rewards, np.ones(10)/10, mode='valid'), label="Dropout + Thompson")
    plt.title("Average Reward Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "rl_dropout_vs_epsilon.png"))
    plt.show()


if __name__ == "__main__":
    run_experiment()
