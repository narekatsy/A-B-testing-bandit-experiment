# Multi-Armed Bandit Experiment - A/B Testing

This project simulates multi-armed bandit algorithms to compare the performance of two popular strategies: **Epsilon-Greedy** and **Thompson Sampling**. It includes an A/B testing setup where rewards are simulated for multiple bandits and the algorithms explore and exploit these bandits to maximize rewards over a series of trials.

## Files
- **Bandit.py**: Contains the implementation of the `Bandit` abstract class, `EpsilonGreedy` class, `ThompsonSampling` class, and the `Visualization` class for plotting the results. Also includes the `comparison` method to run and compare the two algorithms.
- **main.py**: The entry point to run the experiment, execute the comparison of Epsilon-Greedy and Thompson Sampling, and visualize the results.
- **requirements.txt**: Lists the required Python libraries to run the experiment.

## How to Use
Clone the repository

```bash
git clone https://github.com/narekatsy/A-B-testing-bandit-experiment.git
cd A-B-testing-bandit-experiment
```

Install the dependencies by running:

```bash
pip install -r requirements.txt
```

Run the experiment in `main.py`
```bash
python main.py
```

## Outputs
After running the experiment, two CSV files and two visualizations will be generated.

### CSV files
These files contain the following columns:
- Bandit: The bandit (advertisement) chosen at each step.
- Reward: The reward received from the chosen bandit at each step.
- Algorithm: The algorithm used for that particular step (either "EpsilonGreedy" or "ThompsonSampling").

### Visualizations
- Plot 1: Compares the learning process (cumulative rewards) of both algorithms (Epsilon-Greedy vs Thompson Sampling) over time.
- Plot 2: Compares the cumulative regrets of both algorithms.

Cumulative reward shows the total reward accumulated by each algorithm.
Cumulative regret shows the difference between the best possible reward and the actual reward achieved.
