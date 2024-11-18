"""
Simulates multi-armed bandit algorithms for A/B testing.

Includes:
- Bandit (abstract class) for common structure.
- EpsilonGreedy: Implements epsilon-greedy algorithm.
- ThompsonSampling: Implements Thompson Sampling algorithm using Beta distribution.
- Visualization: Methods for comparing algorithm performance visually.
- comparison: A method for running and comparing the performance of EpsilonGreedy and ThompsonSampling algorithms over a specified number of trials.
"""

############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Bandit(ABC):
    def __init__(self, p):
        self.p = p
        self.n = len(p)
        self.counts = [0] * self.n
        self.rewards = [0] * self.n

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """
    Epsilon Greedy algorithm for multi-armed bandit problems.
    Chooses bandits using an epsilon-greedy strategy where it explores with probability epsilon,
    and exploits with probability 1 - epsilon by selecting the bandit with the highest estimated value.
    """
    def __init__(self, p, epsilon):
        """
        Initialize the EpsilonGreedy algorithm.
        
        Args:
        p (list): True probabilities (means) of each bandit's reward.
        epsilon (float): Probability of exploration (random action).
        """
        super().__init__(p)
        self.epsilon = epsilon  # Exploration probability
        self.total_reward = 0  # Total reward collected
        self.num_trials = 0  # Number of trials performed
        self.reward_list = []  # To store rewards per trial
        self.bandit_list = []  # To store the chosen bandits per trial
        self.algo_list = []  # To store which algorithm was used
        self.q_values = [0] * len(p)  # Estimated values for each bandit (arms)
        self.num_pulls = [0] * len(p)  # Number of times each bandit was pulled
        self.bandit_probabilities = p  # True reward probabilities for each bandit

    def __repr__(self):
        """
        String representation of the EpsilonGreedy instance.
        
        Returns:
        str: A string summarizing the current state of the algorithm.
        """
        return f"EpsilonGreedy(epsilon={self.epsilon}, q_values={self.q_values})"

    def pull(self):
        """
        Choose a bandit (arm) based on the epsilon-greedy strategy:
        - With probability epsilon, choose a random bandit (explore).
        - With probability 1 - epsilon, choose the bandit with the highest estimated value (exploit).
        
        Returns:
        int: The index of the chosen bandit.
        """
        if random.random() < self.epsilon:
            return random.randint(0, len(self.bandit_probabilities) - 1)  # Exploration
        else:
            return np.argmax(self.q_values)  # Exploitation

    def update(self, bandit_chosen, reward):
        """
        Update the estimated value (Q-value) for the chosen bandit using the running average formula.
        
        Args:
        bandit_chosen (int): The index of the chosen bandit.
        reward (float): The reward received after pulling the bandit.
        """
        self.num_pulls[bandit_chosen] += 1
        self.q_values[bandit_chosen] += (reward - self.q_values[bandit_chosen]) / self.num_pulls[bandit_chosen]

        # Record the reward and bandit chosen
        self.reward_list.append(reward)
        self.bandit_list.append(bandit_chosen)
        self.algo_list.append("EpsilonGreedy")

        # Update the total reward
        self.total_reward += reward
        self.num_trials += 1

        # Optional: Decay epsilon after each trial
        self.epsilon = self.epsilon / (1 + self.num_trials)

    def experiment(self, num_trials):
        """
        Run the experiment for a specified number of trials.
        At each trial, pull a bandit, observe the reward, and update the algorithm's estimates.
        
        Args:
        num_trials (int): The number of trials to run.
        """
        for _ in range(num_trials):
            bandit_chosen = self.pull()
            reward = random.gauss(self.bandit_probabilities[bandit_chosen], 1)  # Simulate reward with noise
            self.update(bandit_chosen, reward)

    def report(self):
        """
        Generate and display a report including the average reward, cumulative regret, and save results to CSV.
        Also prints the average reward and cumulative regret to the console.
        """
        # Ensure the lists have the same length before creating the DataFrame
        if len(self.bandit_list) == len(self.reward_list) == len(self.algo_list):
            df = pd.DataFrame({
                'Bandit': self.bandit_list,
                'Reward': self.reward_list,
                'Algorithm': self.algo_list
            })

            # Save the results to a CSV file
            df.to_csv('epsilon_greedy_results.csv', index=False)

            # Print the average reward and average regret
            average_reward = sum(self.reward_list) / len(self.reward_list)
            print(f"Average Reward (EpsilonGreedy): {average_reward:.2f}")

            # Cumulative regret calculation
            cumulative_regret = sum([max(self.bandit_probabilities) - r for r in self.reward_list])
            print(f"Cumulative Regret (EpsilonGreedy): {cumulative_regret:.2f}")
        else:
            print("Error: Lists have different lengths!")


#--------------------------------------#

class ThompsonSampling(Bandit):
    """
    Thompson Sampling algorithm for multi-armed bandit problems.
    Selects bandits using Bayesian inference (sampling from Beta distributions representing bandit rewards).
    """
    def __init__(self, p, precision=1):
        """
        Initialize the Thompson Sampling algorithm with prior beliefs using Beta distribution.
        
        Args:
        p (list): True probabilities (means) of each bandit's reward.
        precision (int): The precision for the prior belief (controls the shape of Beta distributions).
        """
        super().__init__(p)
        self.precision = precision  # Prior precision (used for Beta distribution)
        self.total_reward = 0  # Total reward accumulated
        self.num_trials = 0  # Number of trials
        self.reward_list = []  # List of rewards per trial
        self.bandit_list = []  # List of bandits chosen per trial
        self.algo_list = []  # List of algorithms used per trial
        self.alpha = [self.precision] * len(p)  # Alpha parameter for Beta distribution (successes)
        self.beta = [self.precision] * len(p)   # Beta parameter for Beta distribution (failures)
        self.bandit_probabilities = p  # True reward probabilities for each bandit

    def __repr__(self):
        """
        String representation of the ThompsonSampling instance.
        
        Returns:
        str: A string summarizing the current state of the algorithm.
        """
        return f"ThompsonSampling(precision={self.precision}, alpha={self.alpha}, beta={self.beta})"

    def pull(self):
        """
        Use Thompson Sampling to choose a bandit:
        - For each bandit, sample from its Beta distribution (representing its reward probability).
        - Choose the bandit with the highest sample.
        
        Returns:
        int: The index of the bandit chosen based on Thompson Sampling.
        """
        sampled_values = [random.betavariate(self.alpha[i], self.beta[i]) for i in range(len(self.bandit_probabilities))]
        return np.argmax(sampled_values)

    def update(self, bandit_chosen, reward):
        """
        Update the Beta distribution parameters (alpha and beta) for the chosen bandit based on the reward.
        
        Args:
        bandit_chosen (int): The index of the chosen bandit.
        reward (float): The reward received after pulling the bandit.
        """
        if reward > 0:
            self.alpha[bandit_chosen] += 1  # Success, increase alpha
        else:
            self.beta[bandit_chosen] += 1   # Failure, increase beta

        # Record the reward and bandit chosen
        self.reward_list.append(reward)
        self.bandit_list.append(bandit_chosen)
        self.algo_list.append("ThompsonSampling")

        # Update the total reward
        self.total_reward += reward
        self.num_trials += 1

    def experiment(self, num_trials):
        """
        Run the experiment for a specified number of trials.
        At each trial, pull a bandit, observe the reward, and update the algorithm's estimates.
        
        Args:
        num_trials (int): The number of trials to run.
        """
        for _ in range(num_trials):
            bandit_chosen = self.pull()
            reward = random.gauss(self.bandit_probabilities[bandit_chosen], 1)  # Simulate reward with noise
            reward = 1 if reward > 0 else 0  # Binomial reward (1 for success, 0 for failure)
            self.update(bandit_chosen, reward)

    def report(self):
        """
        Generate and display a report including the average reward, cumulative regret, and save results to CSV.
        Also prints the average reward and cumulative regret to the console.
        """
        # Ensure the lists have the same length before creating the DataFrame
        if len(self.bandit_list) == len(self.reward_list) == len(self.algo_list):
            df = pd.DataFrame({
                'Bandit': self.bandit_list,
                'Reward': self.reward_list,
                'Algorithm': self.algo_list
            })

            # Save the results to a CSV file
            df.to_csv('thompson_sampling_results.csv', index=False)

            # Print the average reward and average regret
            average_reward = sum(self.reward_list) / len(self.reward_list)
            print(f"Average Reward (ThompsonSampling): {average_reward:.2f}")

            # Cumulative regret calculation
            cumulative_regret = sum([max(self.bandit_probabilities) - r for r in self.reward_list])
            print(f"Cumulative Regret (ThompsonSampling): {cumulative_regret:.2f}")
        else:
            print("Error: Lists have different lengths!")


#--------------------------------------#

class Visualization:
    """
    A class to handle the plotting of results for comparison between different algorithms.
    """
    def plot1(self, epsilon_greedy_rewards, thompson_sampling_rewards):
        """
        Plot learning curves for both EpsilonGreedy and ThompsonSampling algorithms.
        
        Args:
        epsilon_greedy_rewards (list): List of rewards for EpsilonGreedy algorithm.
        thompson_sampling_rewards (list): List of rewards for ThompsonSampling algorithm.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(epsilon_greedy_rewards, label="Epsilon-Greedy")
        plt.plot(thompson_sampling_rewards, label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.title("Learning Process: Epsilon-Greedy vs Thompson Sampling")
        plt.show()

    def plot2(self, epsilon_greedy_cumulative_rewards, thompson_sampling_cumulative_rewards):
        """
        Compare cumulative rewards and cumulative regrets for both algorithms.
        
        Args:
        epsilon_greedy_cumulative_rewards (list): List of cumulative rewards for EpsilonGreedy.
        thompson_sampling_cumulative_rewards (list): List of cumulative rewards for ThompsonSampling.
        """
        # Plot cumulative reward comparison
        plt.figure(figsize=(12, 6))
        plt.plot(epsilon_greedy_cumulative_rewards, label="Epsilon-Greedy Cumulative Reward")
        plt.plot(thompson_sampling_cumulative_rewards, label="Thompson Sampling Cumulative Reward")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.title("Cumulative Reward Comparison")
        plt.show()

        # Plot cumulative regret comparison
        epsilon_greedy_regret = [max(epsilon_greedy_cumulative_rewards) - r for r in epsilon_greedy_cumulative_rewards]
        thompson_sampling_regret = [max(thompson_sampling_cumulative_rewards) - r for r in thompson_sampling_cumulative_rewards]
        
        plt.figure(figsize=(12, 6))
        plt.plot(epsilon_greedy_regret, label="Epsilon-Greedy Regret")
        plt.plot(thompson_sampling_regret, label="Thompson Sampling Regret")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.title("Cumulative Regret Comparison")
        plt.show()

#--------------------------------------#

def comparison(Bandit_Reward, epsilon, precision, num_trials):
    """
    Run an experiment comparing the EpsilonGreedy and ThompsonSampling algorithms.
    
    Args:
    Bandit_Reward (list): True reward probabilities for the bandits.
    epsilon (float): Exploration probability for EpsilonGreedy.
    precision (int): Precision parameter for ThompsonSampling.
    num_trials (int): Number of trials to run.
    
    Returns:
    tuple: Two DataFrames containing the results of the EpsilonGreedy and ThompsonSampling experiments.
    """
    # Run EpsilonGreedy
    epsilon_greedy = EpsilonGreedy(Bandit_Reward, epsilon)
    epsilon_greedy.experiment(num_trials)
    epsilon_greedy.report()

    # Run ThompsonSampling
    thompson_sampling = ThompsonSampling(Bandit_Reward, precision)
    thompson_sampling.experiment(num_trials)
    thompson_sampling.report()

    # Read results from CSVs
    df_eg = pd.read_csv('epsilon_greedy_results.csv')
    df_ts = pd.read_csv('thompson_sampling_results.csv')

    return df_eg, df_ts

#--------------------------------------#

if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
