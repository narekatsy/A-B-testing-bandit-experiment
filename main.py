import pandas as pd
import matplotlib.pyplot as plt
from Bandit import comparison

def plot_comparison(df_eg, df_ts):
    # Plotting cumulative rewards for both algorithms
    plt.figure(figsize=(12, 6))

    # Epsilon Greedy cumulative reward
    epsilon_greedy_cumsum = df_eg.groupby('Algorithm')['Reward'].cumsum().values
    plt.plot(epsilon_greedy_cumsum, label="EpsilonGreedy", color='blue')

    # Thompson Sampling cumulative reward
    thompson_sampling_cumsum = df_ts.groupby('Algorithm')['Reward'].cumsum().values
    plt.plot(thompson_sampling_cumsum, label="ThompsonSampling", color='red')

    plt.xlabel('Trials')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward Comparison: EpsilonGreedy vs ThompsonSampling')
    plt.legend()
    plt.grid(True)
    plt.show()

    #---------------------------------

    # Plotting cumulative regret for both algorithms
    plt.figure(figsize=(12, 6))

    # Epsilon Greedy cumulative regret
    max_reward = max(df_eg['Reward'].max(), df_ts['Reward'].max())
    epsilon_greedy_regret = (max_reward - df_eg['Reward']).cumsum()
    plt.plot(epsilon_greedy_regret, label="EpsilonGreedy", color='blue')

    # Thompson Sampling cumulative regret
    thompson_sampling_regret = (max_reward - df_ts['Reward']).cumsum()
    plt.plot(thompson_sampling_regret, label="ThompsonSampling", color='red')

    plt.xlabel('Trials')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret Comparison: EpsilonGreedy vs ThompsonSampling')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    Bandit_Reward = [1, 2, 3, 4]
    epsilon = 0.1  # Epsilon for epsilon-greedy
    precision = 1  # Precision for Thompson Sampling
    num_trials = 20000  # Number of trials

    df_eg, df_ts = comparison(Bandit_Reward, epsilon, precision, num_trials)

    plot_comparison(df_eg, df_ts)

if __name__ == '__main__':
    main()
