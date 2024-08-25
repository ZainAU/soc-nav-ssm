import matplotlib.pyplot as plt
import numpy as np
import re

# Define the path to the file
folder_name = 'output_rollout_mamba_2000RL_episodes'
# file_path = '/CrowdNav/crowd_nav/data/' + folder_name + '/output.log'
file_path = 'output-sarl-truncated.log'
# CrowdNav\crowd_nav\data\output_rollout_mamba_2000RL_episodes\output.log

# Initialize lists to store episode numbers and total rewards
episodes = []
rewards = []

# Define the regex pattern to match the relevant lines
pattern = r'INFO: TRAIN in episode (\d+) .* total reward: ([\d\.\-]+)'
# pattern = r'INFO: TRAIN in episode (\d+) .* total reward: ([\d\.\-]+)\)'

# Open the file and process each line
with open(file_path, 'r') as file:
    for line in file:
        match = re.search(pattern, line)
        if match:
            # print(match)
            episodes.append(int(match.group(1)))
            rewards.append(float(match.group(2)))


def plot_simple():
    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, marker='o', linestyle='-')
    plt.xlabel('Episode Number')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs Episode Number (SARL)')
    plt.grid(True)
    plt.show()


def plot_average_rewards(episodes, rewards, window_size=10):
    # Calculate the average reward over the specified window size
    avg_rewards = [np.mean(rewards[i:i+window_size])
                   for i in range(0, len(rewards), window_size)]
    avg_episodes = list(range(1, len(avg_rewards) + 1))

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(avg_episodes, avg_rewards, marker='o', linestyle='-')
    plt.xlabel('Average Episode Number (each point represents 10 episodes)')
    plt.ylabel('Average Total Reward')
    plt.title('Average Total Reward vs Episode Number')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_average_rewards(episodes, rewards)
