import copy
import random
import numpy as np
import matplotlib.pyplot as plt

# Constants
NUM_EPISODES = 5000 # N
NUM_STEPS = 200 # M
GAMMA = 0.9
ETA = 0.2
EPSILON_INIT = 0.1
EPSILON_DECREMENT = 0.01


CAN_VALUE = 2
EMPTY_VALUE = 0
WALL_VALUE = 1

# GRID size is 10x10 with wall padding (+2 rows and columns)
GRID_H = 12
GRID_W = 12
GRID = np.ones((GRID_W, GRID_H), dtype=np.int)
GRID[1:-1,1:-1] = EMPTY_VALUE

POSSIBLE_ACTIONS = ["move_up", "move_left", "move_down", "move_right", "pickup_can"]


class Robby:
    def __init__(self, grid):
        self.pos_x = 0
        self.pos_y = 0
        self.grid = grid

    # Sensor
    def observe_state(self):
        # list with current, up, left, down and right. return 'empty', 'can', or 'wall' for each state
        return [self.grid[self.pos_x][self.pos_y], # Current
                self.grid[self.pos_x][self.pos_y - 1], # Up
                self.grid[self.pos_x - 1][self.pos_y], # Left
                self.grid[self.pos_x][self.pos_y + 1], # Down
                self.grid[self.pos_x + 1][self.pos_y]] # Right
        
    # Actuators
    def move_up(self):
        nextMove = self.pos_y - 1
        if nextMove <= 0:
            return False
        self.pos_y -= 1
        return True

    def move_down(self):
        nextMove = self.pos_y + 1
        if nextMove >= GRID_H - 1:
            return False
        self.pos_y += 1
        return True

    def move_left(self):
        nextMove = self.pos_x - 1
        if nextMove <= 0:
            return False
        self.pos_x -= 1
        return True

    def move_right(self):
        nextMove = self.pos_x + 1
        if nextMove >= GRID_W - 1:
            return False
        self.pos_x += 1
        return True

    def collect_can(self):
        if self.grid[self.pos_x][self.pos_y] == CAN_VALUE:
            # If there is a can, remove it from the grid
            self.grid[self.pos_x][self.pos_y] = EMPTY_VALUE
            return True
        return False


class Solver:
    def __init__(self):
        self.robby = Robby(GRID)
        self.grid = GRID
        self.test_average = 0.0
        self.test_std_dev = 0.0
        self.training_reward_list = []
        self.testing_reward_list = []
        self.episode_reward = 0
        self.total_rewards_per_episode = []

        self.Q_matrix = {} # 5 actions and 3 possible states (empty, wall, can), making number of possible states 3^5=243
        self.epsilon = EPSILON_INIT

    def add_cans(self):
        for i in range(1, GRID_W - 2):
            for j in range(1, GRID_H - 2):
                if random.randint(1, 10) < 5:
                    self.grid[i][j] = CAN_VALUE

    def initialize_grid(self):
        self.grid = copy.deepcopy(GRID)
        self.add_cans()

    def initialize_robby(self):
        self.robby.pos_x = random.randint(1, 10)
        self.robby.pos_y = random.randint(1, 10)
        self.robby.reward_total = 0
        self.robby.grid = self.grid

    def choose_action(self, s_t):
        # Epsilon-greedy action selection
        n = random.uniform(0.0, 1.0)
        if n < self.epsilon:
            action = random.choice(POSSIBLE_ACTIONS)
        else:
            current_Q = list(self.Q_matrix[s_t])
            index = current_Q.index(max(self.Q_matrix[s_t]))
            action = POSSIBLE_ACTIONS[index]

        return action

    def get_step_reward(self, a_t):
        # Calculate reward depending on the action
        if a_t == "move_up":
            if self.robby.move_up():
                return 0 # Move to empty cell
            return -5 # Hit wall
        if a_t == "move_left":
            if self.robby.move_left():
                return 0 # Move to empty cell
            return -5 # Hit wall
        if a_t == "move_down":
            if self.robby.move_down():
                return 0 # Move to empty cell
            return -5 # Hit wall
        if a_t == "move_right":
            if self.robby.move_right():
                return 0 # Move to empty cell
            return -5 # Hit wall
        if a_t == "pickup_can":
            if self.robby.collect_can():
                return 10 # Found can
            return -1 # No can
        return 0

    def train(self):
        for i in range(0, NUM_EPISODES):
            print("Training Episode: " + str(i))
            # Initialize everything except Q Matrix
            self.initialize_grid()
            self.initialize_robby()
            # self.print_grid()
            # self.print_robby_pos()
 
            for j in range(0, NUM_STEPS):
                # Observe Robby's current state s_t
                s_t = tuple(self.robby.observe_state())
                if s_t not in self.Q_matrix:
                    self.Q_matrix[s_t] = np.zeros(5, dtype=np.float)

                #Choose action based on epsilon-greedy policy
                a_t = self.choose_action(s_t)

                # Perform the action and get reward
                r_t = self.get_step_reward(a_t)
                self.episode_reward += r_t

                # Observe the new state
                s_t1 = tuple(self.robby.observe_state())
                if s_t1 not in self.Q_matrix:
                    self.Q_matrix[s_t1] = np.zeros(5, dtype=np.float)

                # Update the Q matrix with the new state and new action
                action_index = POSSIBLE_ACTIONS.index(a_t)
                self.Q_matrix[s_t][action_index] = self.Q_matrix[s_t][action_index] + \
                                                   ETA * (r_t + GAMMA * max(self.Q_matrix[s_t1]) \
                                                   - self.Q_matrix[s_t][action_index])

            # Decrease epsilon every 50 epochs
            if i % 50 == 0:
                if self.epsilon != 0:
                    self.epsilon -= EPSILON_DECREMENT

            # Plot every 100 epochs
            if i % 100 == 0:
                self.training_reward_list.append(self.episode_reward)
            self.episode_reward = 0

        self.plot(self.training_reward_list)


    def test(self):
        self.epsilon = EPSILON_INIT
        self.episode_reward = 0

        # Do the same as training but without touching the Q matrix, only obtain the reward.
        for i in range(0, NUM_EPISODES):
            print("Testing Episode: " + str(i))

            self.initialize_grid()
            self.initialize_robby()

            for j in range(0, NUM_STEPS):
                # Observe Robby's current state s_t
                s_t = tuple(self.robby.observe_state())

                #Choose action based on epsilon-greedy policy
                a_t = self.choose_action(s_t)

                # Perform the action and get reward
                r_t = self.get_step_reward(a_t)
                self.episode_reward += r_t

            # Add the episode rewards
            self.total_rewards_per_episode.append(self.episode_reward)

            # Plot every 100 episodes
            if i % 100 == 0:
                self.testing_reward_list.append(self.episode_reward)
            self.episode_reward = 0

        mean, std = self.gaussian_values()
        print('Test-Average:  ' + str(mean))
        print('Test-Standard-Deviation: ' + str(std))
        self.plot(self.testing_reward_list)

    def gaussian_values(self):
        mean = np.sum(self.total_rewards_per_episode) / NUM_EPISODES
        std = np.std(self.total_rewards_per_episode)

        return mean, std

    def plot(self, reward):
        plt.plot(reward)
        plt.ylabel('Total Reward Per Episode')
        plt.xlabel('Episodes * 100')
        plt.show()

    def print_grid(self):
        print(np.matrix(self.grid))

    def print_robby_pos(self):
        print("Robby's Position (row, col): (" + str(self.robby.pos_x) + ", " + str(self.robby.pos_y) + ')\n')

s = Solver()
s.train()
s.test()