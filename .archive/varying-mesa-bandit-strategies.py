import mesa
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from ipywidgets import interact, IntSlider


class ArmAgent(mesa.Agent):
    """An agent representing an arm in the multi-armed bandit problem."""
    
    def __init__(self, unique_id, model, initial_mean, initial_std, variation_func):
        super().__init__(unique_id, model)
        self.initial_mean = initial_mean
        self.initial_std = initial_std
        self.variation_func = variation_func
    
    def get_mean(self):
        """Get the current mean based on the model's current step."""
        return self.variation_func(self.initial_mean, self.model.schedule.steps)
    
    def get_std(self):
        """Get the current standard deviation."""
        return self.initial_std
    
    def pull(self):
        """Pull the arm and return a reward."""
        return np.random.normal(self.get_mean(), self.get_std())

def linear_variation(initial_mean, t):
    return initial_mean + 0.01 * t

def sinusoidal_variation(initial_mean, t):
    return initial_mean + np.sin(t / 10) * 0.5


class BanditAgent(mesa.Agent):
    """An agent representing the bandit in the multi-armed bandit problem."""
    
    def __init__(self, unique_id, model, strategy='epsilon_greedy', epsilon=0.1):
        super().__init__(unique_id, model)
        self.strategy = strategy
        self.epsilon = epsilon
        self.values = {arm.unique_id: 0 for arm in self.model.arms}
        self.counts = {arm.unique_id: 0 for arm in self.model.arms}
        self.cumulative_reward = 0
        self.arm_selections = []
    
    def select_arm_random(self):
        """Select an arm randomly."""
        return self.random.choice(self.model.arms)
    
    def select_arm_greedy(self):
        """Select the arm with the highest estimated value."""
        return max(self.model.arms, key=lambda arm: self.values[arm.unique_id])
    
    def select_arm_epsilon_greedy(self):
        """Select an arm using epsilon-greedy strategy."""
        if self.random.random() < self.epsilon:
            return self.select_arm_random()
        else:
            return self.select_arm_greedy()
    
    def select_arm(self):
        """Select an arm based on the current strategy."""
        if self.strategy == 'random':
            return self.select_arm_random()
        elif self.strategy == 'greedy':
            return self.select_arm_greedy()
        elif self.strategy == 'epsilon_greedy':
            return self.select_arm_epsilon_greedy()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def update(self, chosen_arm, reward):
        """Update the estimated value of the chosen arm."""
        arm_id = chosen_arm.unique_id
        self.counts[arm_id] += 1
        n = self.counts[arm_id]
        value = self.values[arm_id]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_id] = new_value
        self.cumulative_reward += reward
    
    def step(self):
        """Perform one step of the bandit algorithm."""
        chosen_arm = self.select_arm()
        reward = chosen_arm.pull()
        self.update(chosen_arm, reward)
        self.arm_selections.append(chosen_arm.unique_id) 


class MABModel(mesa.Model):
    """A model for the multi-armed bandit problem."""
    
    def __init__(self, arm_params, bandit_strategy='epsilon_greedy'):
        self.num_arms = len(arm_params)
        self.schedule = mesa.time.RandomActivation(self)
        
        # Create arms based on the provided parameters
        self.arms = []
        for i, (initial_mean, initial_std, variation_func) in enumerate(arm_params):
            arm = ArmAgent(i, self, initial_mean, initial_std, variation_func)
            self.arms.append(arm)
            self.schedule.add(arm)
        
        # Create bandit
        self.bandit = BanditAgent(self.num_arms, self, strategy=bandit_strategy)
        self.schedule.add(self.bandit)
        
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Cumulative Reward": lambda m: m.bandit.cumulative_reward,
                "Strategy": lambda m: m.bandit.strategy,
                "Arm Selections": lambda m: m.bandit.arm_selections
            },
            agent_reporters={}
        )
    
    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        self.datacollector.collect(self)


def visualize_true_distributions(arms, max_time=1000):
    plt.figure(figsize=(12, 6))
    x = np.linspace(-5, 5, 1000)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(arms)))
    
    def update(time):
        plt.clf()
        for i, (arm, color) in enumerate(zip(arms, colors)):
            mean = arm.variation_func(arm.initial_mean, time)
            std = arm.get_std()
            y = norm.pdf(x, mean, std)
            plt.plot(x, y, label=f'Arm {i} (μ={mean:.2f}, σ={std:.2f})', color=color)
            
            # Add vertical line at the mean
            plt.axvline(x=mean, color=color, linestyle='--', alpha=0.7)
        
        plt.title(f"Reward Distributions for Each Arm (Time: {time})")
        plt.xlabel("Reward")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 0.5)
        plt.show()
    
    # Create an interactive slider
    from ipywidgets import interact, IntSlider
    time_slider = IntSlider(min=0, max=max_time, step=1, value=0, description='Time:')
    interact(update, time=time_slider)


# Define arms with different initial means, standard deviations, and variation functions
arm_params = [
    (0.0, 1, linear_variation),
    (0.5, 1, sinusoidal_variation),
    (1.0, 1, lambda m, t: m),
    (1.5, 1, lambda m, t: m + 0.005 * t * np.sin(t / 5)),
    (2.0, 1, lambda m, t: m - 0.02 * t)
]


# Visualize the true reward distributions
visualize_true_distributions([ArmAgent(i, None, *params) for i, params in enumerate(arm_params)])


def visualize_arm_selection_frequencies(all_selections, arms, max_time=1000):
    plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(arms)))
    
    def update(time):
        plt.clf()
        selections_up_to_time = {strategy: selections[:time+1] for strategy, selections in all_selections.items()}
        
        x = np.arange(len(arms))
        bar_width = 0.25
        multiplier = 0

        for strategy, selections in selections_up_to_time.items():
            arm_pulls = [selections.count(arm.unique_id) for arm in arms]
            offset = bar_width * multiplier
            rects = plt.bar(x + offset, arm_pulls, bar_width, label=strategy)
            plt.bar_label(rects, padding=3)
            multiplier += 1

        plt.title(f"Arm Selection Frequencies (Time: {time})")
        plt.xlabel("Arm")
        plt.ylabel("Number of Pulls")
        plt.xticks(x + bar_width, [f"Arm {i}" for i in range(len(arms))])
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Create an interactive slider
    time_slider = IntSlider(min=0, max=max_time, step=1, value=0, description='Time:')
    interact(update, time=time_slider)


# Simulation parameters
num_steps = 1000
strategies = ['random', 'greedy', 'epsilon_greedy']


# Run simulations for each strategy
results = {}
models = {}
all_selections = {}

for strategy in strategies:
    model = MABModel(arm_params, bandit_strategy=strategy)
    for _ in range(num_steps):
        model.step()
    
    results[strategy] = model.datacollector.get_model_vars_dataframe()
    models[strategy] = model
    all_selections[strategy] = model.bandit.arm_selections


# Visualize arm selection frequencies
visualize_arm_selection_frequencies(all_selections, models[strategies[0]].arms, max_time=num_steps-1)


# Plot cumulative reward over time
plt.figure(figsize=(12, 6))
for strategy, data in results.items():
    plt.plot(data['Cumulative Reward'], label=strategy)

plt.title(f"Cumulative Reward Over Time ({len(arm_params)} arms, {num_steps} steps)")
plt.xlabel("Steps")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)
plt.show()


# Plot average reward over time
plt.figure(figsize=(12, 6))
for strategy, data in results.items():
    average_reward = data['Cumulative Reward'] / (np.arange(len(data)) + 1)
    plt.plot(average_reward, label=strategy)

plt.title(f"Average Reward Over Time ({len(arm_params)} arms, {num_steps} steps)")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend()
plt.grid(True)
plt.show()


# Print final arm values and counts for each strategy
for strategy, model in models.items():
    print(f"\nFinal arm values and counts for {strategy} strategy:")
    bandit = model.schedule.agents[-1]
    for arm in model.arms:
        print(f"Arm {arm.unique_id}: Final mean = {arm.get_mean():.2f}, "
              f"Estimated value = {bandit.values[arm.unique_id]:.2f}, "
              f"Pull count = {bandit.counts[arm.unique_id]}")
    print(f"Total reward: {bandit.cumulative_reward:.2f}")
