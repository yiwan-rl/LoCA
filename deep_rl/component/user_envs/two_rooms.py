import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete


class TwoRooms:

    def __init__(self):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c == 'w' else 0, line)) for line in layout.splitlines()])

        # Four possible actions
        # 0: UP
        # 1: DOWN
        # 2: LEFT
        # 3: RIGHT
        self.observation_space = Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32)
        self.action_space = Discrete(4)
        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.reward_range = np.array([0, 1])
        self.tostate = {}
        statenum = 0
        for i in range(7):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1
        self.tocell = {v: k for k, v in self.tostate.items()}

        self.goal = 42  # East doorway
        self.init_states = list(range(np.sum(self.occupancy == 0)))
        self.init_states.remove(self.goal)
        # self.init_states = list(range(1))
        self.eps_steps = 0

    def render(self, show_goal=True):
        current_grid = np.array(self.occupancy)
        current_grid[self.current_cell[0], self.current_cell[1]] = -1
        if show_goal:
            goal_cell = self.tocell[self.goal]
            current_grid[goal_cell[0], goal_cell[1]] = -1
        return current_grid

    def reset(self):
        self.eps_steps = 0
        state = self.rng.choice(self.init_states)
        self.current_cell = self.tocell[state]
        obs = self.to_obs(self.current_cell)
        return obs

    def check_available_cells(self, cell):
        available_cells = []

        for action in range(self.action_space.n):
            next_cell = tuple(cell + self.directions[action])

            if not self.occupancy[next_cell]:
                available_cells.append(next_cell)

        return available_cells

    def step(self, action):
        '''
        Takes a step in the environment with 2/3 probability. And takes a step in the
        other directions with probability 1/3 with all of them being equally likely.
        '''

        next_cell = tuple(self.current_cell + self.directions[action])

        if not self.occupancy[next_cell]:

            # if self.rng.uniform() < 1 / 3:
            #     available_cells = self.check_available_cells(self.current_cell)
            #     self.current_cell = available_cells[self.rng.randint(len(available_cells))]
            #
            # else:
            #     self.current_cell = next_cell
            self.current_cell = next_cell

        state = self.tostate[self.current_cell]

        # When goal is reached, it is done
        done = state == self.goal

        obs = self.to_obs(self.current_cell)

        info = {}

        self.eps_steps += 1
        
        reward = done
        if self.eps_steps == 100:
            done = True

        return obs, reward, done, info

    # def to_obs(self, cell):
    #     obs = np.zeros(91)
    #     obs[cell[0] * 13 + cell[1]] = 1
    #     return obs
    
    def to_obs(self, cell):
        obs = np.zeros(2)
        obs[0] = cell[0]
        obs[1] = cell[1]
        # if cell[1] <= 6:
        #     obs[0] = cell[0] / 10.
        #     obs[1] = cell[1] / 10.
        #     obs[2] = 1
        # else:
        #     obs[0] = cell[0] / 10.
        #     obs[1] = cell[1] / 10.
        #     obs[3] = 1
        return obs

    def seed(self, seed):
        # Random number generator
        self.rng = np.random.RandomState(seed)