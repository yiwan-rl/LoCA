import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete


class TwoRoomsThreeGoals:

    def __init__(self):
#         layout = """\
# wwwwwwwwwwwww
# wgggggwgggggw
# wgggggwgggggw
# wgggggggggggw
# wgggggwgggggw
# wgggggwgggggw
# wwgwwwwgggggw
# wgggggwwwgwww
# wgggggwgggggw
# wgggggwgggggw
# wgggggggggggw
# wgggggwgggggw
# wwwwwwwwwwwww
# """
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w  g        w
w     w g   w
w     w     w
ww wwww     w
w     www www
w     w     w
w  g  w     w
w         g w
w     w     w
wwwwwwwwwwwww
"""
#         layout = """\
# wwwwwwwwwwwww
# w     w     w
# w     w     w
# w           w
# w     w     w
# w     w     w
# ww wwww     w
# w     www www
# w     w     w
# w     wg    w
# w           w
# w     w     w
# wwwwwwwwwwwww
# """
#         layout = """\
# wwwwwwwwwwwww
# wgggggwgggggw
# wgggggwgggggw
# wgggggggggggw
# wgggggwgggggw
# wgggggwgggggw
# wwwwwwwwwwwww
# """
        self.occupancy = np.array([list(map(lambda c: 1 if c == 'w' else 0, line)) for line in layout.splitlines()])
        self.height, self.width = self.occupancy.shape
        self.goal_occupancy = np.array([list(map(lambda c: 1 if c == 'g' else 0, line)) for line in layout.splitlines()])

        # Four possible actions
        # 0: UP
        # 1: DOWN
        # 2: LEFT
        # 3: RIGHT
        self.observation_space = Box(low=-100.0, high=100.0, shape=(4,), dtype=np.float32)
        self.tabular_dim = self.height * self.width
        self.action_space = Discrete(4)
        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.reward_range = np.array([0, 1])
        self.tostate = {}
        self.goals = []
        
        statenum = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.occupancy[i, j] == 0:
                    if self.goal_occupancy[i, j] == 1:
                        self.goals.append(statenum)
                    self.tostate[(i, j)] = statenum
                    statenum += 1
        self.tocell = {v: k for k, v in self.tostate.items()}
        
        self.init_states = list(range(np.sum(self.occupancy == 0)))
        # for g in self.goals:
        #     self.init_states.remove(g)

        # self.init_states = [self.tostate[(3, 6)]]
        
        self.eps_steps = 0

    def render(self, show_goal=True):
        current_grid = np.array(self.occupancy)
        current_grid[self.current_cell[0], self.current_cell[1]] = -1
        if show_goal:
            goal_cell = self.tocell[self.goals[self.goal_idx]]
            current_grid[goal_cell[0], goal_cell[1]] = -1
        return current_grid

    def reset(self):
        self.eps_steps = 0
        state = self.rng.choice(self.init_states)
        # state = self.rng.choice([0, 2, 4, 10, 14, 20, 24])
        self.goal_idx = self.rng.randint(self.goals.__len__())
        self.current_cell = self.tocell[state]
        obs = self.to_obs(self.current_cell, self.tocell[self.goals[self.goal_idx]])
        return obs
    
    def reset_test(self):
        self.eps_steps = 0
        # state = self.rng.choice(self.init_states)
        state = self.rng.choice([0, 2, 4, 10, 14, 20, 24])
        self.goal_idx = self.rng.randint(self.goals.__len__())
        self.current_cell = self.tocell[state]
        self.start_cell = self.current_cell
        obs = self.to_obs(self.current_cell, self.tocell[self.goals[self.goal_idx]])
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
        done = state == self.goals[self.goal_idx]
        # reward = done
        reward = -0.1

        obs = self.to_obs(self.current_cell, self.tocell[self.goals[self.goal_idx]])

        info = {}

        self.eps_steps += 1
        if self.eps_steps == 100:
            done = True

        return obs, reward, done, info

    def xy_to_tabular(self, xy):
        tabular = np.zeros((xy.shape[0], 169))
        h = xy[:, 0] + (self.height - 1) / 2
        w = xy[:, 1] + (self.width - 1) / 2
        indices = h * self.width + w
        for i in range(xy.shape[0]):
            tabular[i, int(indices[i])] = 1
        return tabular
    
    def to_obs(self, cell, goal_cell):
        obs = np.zeros(4)
        obs[0] = cell[0] - (self.height - 1) / 2
        obs[1] = cell[1] - (self.width - 1) / 2
        obs[2] = goal_cell[0] - (self.height - 1) / 2
        obs[3] = goal_cell[1] - (self.width - 1) / 2
        obs = obs
        # obs[0] = cell[0]
        # obs[1] = cell[1]
        # obs[2] = goal_cell[0]
        # obs[3] = goal_cell[1]

        # if cell[1] <= 6:
        #     obs[0] = cell[0] / 10.
        #     obs[1] = cell[1] / 10.
        #     obs[2] = 0
        # else:
        #     obs[0] = cell[0] / 10.
        #     obs[1] = cell[1] / 10.
        #     obs[2] = 1
        return obs

    def seed(self, seed):
        # Random number generator
        self.rng = np.random.RandomState(seed)


class ChangingGoalFourRooms:
    
    def __init__(self):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     wg    w
w           w
w     w     w
wwwwwwwwwwwww
"""
        
        self.init_new_map(layout)
        self.eps_steps = 0
        self.total_steps = 0
        
    def init_new_map(self, layout):
        self.occupancy = np.array([list(map(lambda c: 1 if c == 'w' else 0, line)) for line in layout.splitlines()])
        self.height, self.width = self.occupancy.shape
        self.goal_occupancy = np.array(
            [list(map(lambda c: 1 if c == 'g' else 0, line)) for line in layout.splitlines()])
    
        # Four possible actions
        # 0: UP
        # 1: DOWN
        # 2: LEFT
        # 3: RIGHT
        self.observation_space = Box(low=-100.0, high=100.0, shape=(4,), dtype=np.float32)
        self.action_space = Discrete(4)
        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.reward_range = np.array([0, 1])
        self.tostate = {}
        self.goals = []
    
        statenum = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.occupancy[i, j] == 0:
                    if self.goal_occupancy[i, j] == 1:
                        self.goals.append(statenum)
                    self.tostate[(i, j)] = statenum
                    statenum += 1
        self.tocell = {v: k for k, v in self.tostate.items()}
    
        self.init_states = list(range(np.sum(self.occupancy == 0)))
    
    def render(self, show_goal=True):
        current_grid = np.array(self.occupancy)
        current_grid[self.current_cell[0], self.current_cell[1]] = -1
        if show_goal:
            goal_cell = self.tocell[self.goals[self.goal_idx]]
            current_grid[goal_cell[0], goal_cell[1]] = -1
        return current_grid
    
    def reset(self):
        self.eps_steps = 0
        state = self.rng.choice(self.init_states)
        self.goal_idx = self.rng.randint(self.goals.__len__())
        self.current_cell = self.tocell[state]
        obs = self.to_obs(self.current_cell, self.tocell[self.goals[self.goal_idx]])
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
        done = state == self.goals[self.goal_idx]
        # reward = done
        reward = -0.1
        obs = self.to_obs(self.current_cell, self.tocell[self.goals[self.goal_idx]])
        
        info = {}
        
        self.eps_steps += 1
        self.total_steps += 1
    
        if self.total_steps == 30000:
            layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w          gw
w     w     w
wwwwwwwwwwwww
"""
            self.init_new_map(layout)
        
        if self.eps_steps == 100:
            done = True
        
        return obs, reward, done, info
    
    # def to_obs(self, cell):
    #     obs = np.zeros(91)
    #     obs[cell[0] * 13 + cell[1]] = 1
    #     # obs[91 + self.goal_idx] = 1
    #     return obs
    
    def to_obs(self, cell, goal_cell):
        obs = np.zeros(4)
        obs[0] = cell[0] - (self.height - 1) / 2
        obs[1] = cell[1] - (self.width - 1) / 2
        obs[2] = goal_cell[0] - (self.height - 1) / 2
        obs[3] = goal_cell[1] - (self.width - 1) / 2
        # if cell[1] <= 6:
        #     obs[0] = cell[0] / 10.
        #     obs[1] = cell[1] / 10.
        #     obs[2] = 0
        # else:
        #     obs[0] = cell[0] / 10.
        #     obs[1] = cell[1] / 10.
        #     obs[2] = 1
        return obs
    
    def seed(self, seed):
        # Random number generator
        self.rng = np.random.RandomState(seed)