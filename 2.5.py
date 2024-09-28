import random
import math
import pandas as pd
import matplotlib.pyplot as plt


class Banditarm:
    def __init__(self, mean, deviation):
        self.mean = mean
        self.deviation = deviation

    def generate_reward(self):
        return random.gauss(self.mean, self.deviation)


class KArmedBandit:
    def __init__(self, n):
        self.bandit_arms = self.generate_testbed(n)

    def generate_testbed(self, n):
        bandit_arms = list()
        for i in range(n):
            bandit_arms.append(Banditarm(random.gauss(), 1.0))
        return bandit_arms

    def reward(self, action):
        return self.bandit_arms[action].generate_reward()

    def optimal_action(self):
        greatest = 0
        i = 0
        for index, arm in enumerate(self.bandit_arms):
            if arm.mean > greatest:
                greatest = arm.mean
                i = index

        return i


class SampleAverage:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.action_values = [0] * 10
        self.action_frequencies = [0] * 10

    def action(self):
        max_val = self.action_values[0]
        optimal_action = 0
        for action, value in enumerate(self.action_values):
            if value > max_val:
                max_val = value
                optimal_action = action

        if random.uniform(0, 1) > 0.95:
            return math.floor(random.uniform(0, 9))
        return optimal_action

    def update(self, action, reward):
        self.action_frequencies[action] += 1
        self.action_values[action] -= (1 / self.action_frequencies[action]) * (self.action_values[action] - reward)


class Runner:
    def __init__(self, agent, environment):
        self.agent = agent
        self.environment = environment
        self.actions = []
        self.rewards = []

    def run_step(self):
        action = self.agent.action()
        reward = self.environment.reward(action)
        self.agent.update(action, reward)
        return action, reward

    def run(self, steps):
        for j in range(steps):
            action, reward = self.run_step()
            self.actions.append(action)
            self.rewards.append(reward)


steps = 10000
runs = 2000
average_reward = [0] * steps
optim_percentage = [0] * steps
for run in range(runs):
    agent = SampleAverage(0.1)
    environment = KArmedBandit(10)

    runner = Runner(agent, environment)
    runner.run(steps)
    average_reward = [average_reward[i] + runner.rewards[i] for i in range(steps)]
    optimal_action = environment.optimal_action()
    is_optim = [1 if i == optimal_action else 0 for i in runner.actions]
    optim_percentage = [optim_percentage[i] + is_optim[i] for i in range(steps)]
    print(run)

average_reward = [x / runs for x in average_reward]
optim_percentage = [(x / runs) * 100 for x in optim_percentage]

plt.figure(1)
plt.plot(average_reward)
plt.ylabel('Average Reward')
plt.xlabel('Steps')
plt.savefig('./average_reward.png')

plt.figure(2)
plt.plot(optim_percentage)
plt.xlabel('Steps')
plt.ylabel('% Optimal action')
plt.savefig('./optimal_percentage.png')
