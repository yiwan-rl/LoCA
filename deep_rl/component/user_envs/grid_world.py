import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete


# class FourRoomsOneGoalOptionsBase:
#
#     def __init__(self, args):
#         #         layout = """\
#         # wwwwwwwwwwwww
#         # wgggggwgggggw
#         # wgggggwgggggw
#         # wgggggggggggw
#         # wgggggwgggggw
#         # wgggggwgggggw
#         # wwgwwwwgggggw
#         # wgggggwwwgwww
#         # wgggggwgggggw
#         # wgggggwgggggw
#         # wgggggggggggw
#         # wgggggwgggggw
#         # wwwwwwwwwwwww
#         # """
#         #         layout = """\
#         # wwwwwwwwwwwww
#         # w     w     w
#         # w     w     w
#         # w  g        w
#         # w     w g   w
#         # w     w     w
#         # ww wwww     w
#         # w     www www
#         # w     w     w
#         # w  g  w     w
#         # w         g w
#         # w     w     w
#         # wwwwwwwwwwwww
#         # """
#         layout_G1 = """\
# wwwwwwwwwwwww
# w     w     w
# w     w     w
# w           w
# w     w     w
# w     w     w
# www www     w
# w     www www
# w     w     w
# w     w g   w
# w           w
# w     w     w
# wwwwwwwwwwwww
# """
#         layout_G2 = """\
# wwwwwwwwwwwww
# w     w     w
# w     w     w
# w           w
# w     w     w
# w     w     w
# www www     w
# w     www www
# w     w     w
# w     w     w
# w     g     w
# w     w     w
# wwwwwwwwwwwww
# """
#         option4 = """\
# wwwwwwwwwwwww
# w>>>>vw     w
# w>>>>vw     w
# w>>>>>      w
# w>>>>^w     w
# w>>>>^w     w
# www^www     w
# w     www www
# w     w     w
# w     w     w
# w           w
# w     w     w
# wwwwwwwwwwwww
# """
#         option5 = """\
# wwwwwwwwwwwww
# w>>v<<w     w
# w>>v<<w     w
# w>>v<<<     w
# w>>v<<w     w
# w>>v<<w     w
# www www     w
# w     www www
# w     w     w
# w     w     w
# w           w
# w     w     w
# wwwwwwwwwwwww
# """
#         option6 = """\
# wwwwwwwwwwwww
# w     wvvvvvw
# w     wvvvvvw
# w      <<<<<w
# w     w^^^^^w
# w     w^^^^^w
# www www^^^^^w
# w     www^www
# w     w     w
# w     w     w
# w           w
# w     w     w
# wwwwwwwwwwwww
# """
#         option7 = """\
# wwwwwwwwwwwww
# w     w>>v<<w
# w     w>>v<<w
# w     >>>v<<w
# w     w>>v<<w
# w     w>>v<<w
# www www>>v<<w
# w     www www
# w     w     w
# w     w     w
# w           w
# w     w     w
# wwwwwwwwwwwww
# """
#         option8 = """\
# wwwwwwwwwwwww
# w     w     w
# w     w     w
# w           w
# w     w     w
# w     w     w
# wwwvwww     w
# wvvvvvwww www
# wvvvvvw     w
# wvvvvvw     w
# w>>>>>      w
# w^^^^^w     w
# wwwwwwwwwwwww
# """
#         option9 = """\
# wwwwwwwwwwwww
# w     w     w
# w     w     w
# w           w
# w     w     w
# w     w     w
# www www     w
# w>>^<<www www
# w>>^<<w     w
# w>>^<<w     w
# w>>^<<<     w
# w>>^<<w     w
# wwwwwwwwwwwww
# """
#         option10 = """\
# wwwwwwwwwwwww
# w     w     w
# w     w     w
# w           w
# w     w     w
# w     w     w
# www www     w
# w     www www
# w     w>>^<<w
# w     w>>^<<w
# w     >>>^<<w
# w     w>>^<<w
# wwwwwwwwwwwww
# """
#         option11 = """\
# wwwwwwwwwwwww
# w     w     w
# w     w     w
# w           w
# w     w     w
# w     w     w
# www www     w
# w     wwwvwww
# w     wvvvvvw
# w     wvvvvvw
# w      <<<<<w
# w     w^^^^^w
# wwwwwwwwwwwww
# """
#         #         layout = """\
#         # wwwwwwwwwwwww
#         # wgggggwgggggw
#         # wgggggwgggggw
#         # wgggggggggggw
#         # wgggggwgggggw
#         # wgggggwgggggw
#         # wwwwwwwwwwwww
#         # """
#         if args is None:
#             layout = layout_G1
#             restart = "FixedRestart"
#         else:
#             if "G1" in args:
#                 layout = layout_G1
#             elif "G2" in args:
#                 layout = layout_G2
#             else:
#                 raise NotImplementedError
#
#             if "FixedRestart" in args:
#                 restart = "FixedRestart"
#             elif "RandomRestart" in args:
#                 restart = "RandomRestart"
#             else:
#                 raise NotImplementedError
#
#         self.occupancy = np.array([list(map(lambda c: 1 if c == 'w' else 0, line)) for line in layout.splitlines()])
#         self.options = []
#         for i in range(4, 12):
#             self.options.append(np.array([list(map(
#                 lambda c: 0 if c == '^' else (
#                     1 if c == 'v' else (
#                         2 if c == '<' else (
#                             3 if c == '>' else -1
#                         )
#                     )
#                 ), line)) for line in eval("option" + str(i)).splitlines()])
#             )
#         self.options = np.array(self.options)
#         self.height, self.width = self.occupancy.shape
#         self.goal_occupancy = np.array(
#             [list(map(lambda c: 1 if c == 'g' else 0, line)) for line in layout.splitlines()]
#         )
#
#         # Four possible actions
#         # 0: UP
#         # 1: DOWN
#         # 2: LEFT
#         # 3: RIGHT
#         self.tostate = {}
#         self.goals = []
#         statenum = 0
#         for i in range(self.height):
#             for j in range(self.width):
#                 if self.occupancy[i, j] == 0:
#                     if self.goal_occupancy[i, j] == 1:
#                         self.goals.append(statenum)
#                     self.tostate[(i, j)] = statenum
#                     statenum += 1
#         self.total_statenum = statenum
#         self.tocell = {v: k for k, v in self.tostate.items()}
#         self.init_states = list(range(np.sum(self.occupancy == 0)))
#
#         self.observation_space = Box(low=-100.0, high=100.0, shape=(self.total_statenum,), dtype=np.float32)
#         self.tabular_dim = self.height * self.width
#         self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
#         self.reward_range = np.array([0, 1])
#         self.metadata = None
#         self.primitive_action_space = Discrete(4)
#
#         if restart == 'FixedRestart':
#             self.init_states = [self.tostate[(6, 3)]]
#         elif restart == 'RandomRestart':
#             for g in self.goals:
#                 self.init_states.remove(g)
#         else:
#             raise NotImplementedError
#
#         self.eps_steps = 0
#         self.last_reset_step = 0
#
#     def render(self, show_goal=True):
#         current_grid = np.array(self.occupancy)
#         current_grid[self.current_cell[0], self.current_cell[1]] = 2
#         if show_goal:
#             goal_cell = self.tocell[self.goals[self.goal_idx]]
#             current_grid[goal_cell[0], goal_cell[1]] = -1
#         return current_grid
#
#     def render_policy(self, optimal_actions):
#         action_num_to_symbol = {0: '^', 1: 'v', 2: '<', 3: '>'}
#         grid = np.array(self.occupancy).astype(str)
#         for i in range(optimal_actions.shape[0]):
#             grid[self.tocell[i]] = action_num_to_symbol[optimal_actions[i]]
#         grid = np.array2string(grid, separator=',', formatter={'str_kind': lambda x: x})
#         return grid
#
#     def render_policy_over_options(self, optimal_options):
#         action_num_to_symbol = {0: '^', 1: 'v', 2: '<', 3: '>'}
#         grid = np.array(self.occupancy).astype(str)
#         for i in range(optimal_options.shape[0]):
#             if optimal_options[i] < 4:
#                 grid[self.tocell[i]] = action_num_to_symbol[optimal_options[i]]
#             else:
#                 grid[self.tocell[i]] = optimal_options[i]
#         grid = np.array2string(grid, separator=',', formatter={'str_kind': lambda x: x})
#         return grid
#
#     def render_policy_over_options_options_only(self, optimal_options):
#         grid = np.array(self.occupancy).astype(str)
#         for i in range(optimal_options.shape[0]):
#             grid[self.tocell[i]] = optimal_options[i]
#         grid = np.array2string(grid, separator=',', formatter={'str_kind': lambda x: x})
#         return grid
#
#     def reset(self):
#         self.eps_steps = 0
#         state = self.rng.choice(self.init_states)
#         # state = self.rng.choice([0, 2, 4, 10, 14, 20, 24])
#         self.goal_idx = self.rng.randint(self.goals.__len__())
#         self.current_cell = self.tocell[state]
#         obs = self.to_obs(self.current_cell)
#         return obs
#
#     def reset_test(self):
#         self.eps_steps = 0
#         # state = self.rng.choice(self.init_states)
#         state = self.rng.choice([0, 2, 4, 10, 14, 20, 24])
#         self.goal_idx = self.rng.randint(self.goals.__len__())
#         self.current_cell = self.tocell[state]
#         self.start_cell = self.current_cell
#         obs = self.to_obs(self.current_cell)
#         return obs
#
#     def check_available_cells(self, cell):
#         available_cells = []
#
#         for action in range(self.primitive_action_space.n):
#             next_cell = tuple(cell + self.directions[action])
#
#             if not self.occupancy[next_cell]:
#                 available_cells.append(next_cell)
#             else:
#                 available_cells.append(cell)
#
#         return available_cells
#
#     def option_action(self, cell, option):
#         raise NotImplementedError
#
#     def option_termination(self, cell, option):
#         raise NotImplementedError
#
#     def get_action_prob(self, cell, option, action):
#         option_action, random_action = self.option_action(cell, option)
#         if random_action:
#             return 0.25
#         if option_action == action:
#             return 1.0
#         else:
#             return 0.0
#
#     def get_all_options_action_probs(self, cell, action):
#         options_probs = []
#         for option in range(self.action_space.n):
#             options_probs.append(self.get_action_prob(cell, option, action))
#         return options_probs
#
#     def get_termination_prob(self, cell, option):
#         term = self.option_termination(cell, option)
#         return term
#
#     def get_all_options_term_probs(self, cell):
#         options_probs = []
#         for option in range(self.action_space.n):
#             options_probs.append(self.get_termination_prob(cell, option))
#         return options_probs
#
#     def step(self, option):
#         '''
# 		Takes a step in the environment with 2/3 probability. And takes a step in the
# 		other directions with probability 1/3 with all of them being equally likely.
# 		'''
#         option, as_action = option
#         option_term = None
#         if as_action:
#             action = option
#             option_term = True
#         else:
#             action, _ = self.option_action(self.current_cell, option)
#
#         assert (0 <= action and action <= 3)
#         # print(action)
#         next_cell = tuple(self.current_cell + self.directions[action])
#
#         option_action_probs = self.get_all_options_action_probs(self.current_cell, action)
#
#         if not self.occupancy[next_cell]:
#             # if self.rng.uniform() < 4. / 16.:
#             #     available_cells = self.check_available_cells(self.current_cell)
#             #     self.current_cell = available_cells[self.rng.randint(len(available_cells))]
#             #
#             # else:
#             #     self.current_cell = next_cell
#             self.current_cell = next_cell
#
#         state = self.tostate[self.current_cell]
#         if option_term is None:
#             option_term = self.option_termination(self.current_cell, option)
#
#         option_term_probs = self.get_all_options_term_probs(self.current_cell)
#         info = {
#             "option_term": option_term, "action": action,
#             "options_action_probs": option_action_probs, "options_term_probs": option_term_probs
#         }
#
#         # When goal is reached, it is done
#         # done = state == self.goals[self.goal_idx]
#         # reward = done
#         # reward = -0.1
#         if state == self.goals[self.goal_idx]:
#             reward = 1.0
#             self.last_reset_step = self.eps_steps
#             obs = self.reset()
#         elif self.eps_steps - self.last_reset_step == 1000:
#             self.last_reset_step = self.eps_steps
#             reward = 0
#             obs = self.reset()
#         else:
#             reward = 0
#             obs = self.to_obs(self.current_cell)
#
#         self.eps_steps += 1
#         done = False
#
#         return obs, reward, done, info
#
#     def to_obs(self, cell):
#         obs = np.zeros(self.total_statenum)
#         obs[self.tostate[cell]] = 1
#         return obs
#
#     def seed(self, seed):
#         # Random number generator
#         self.rng = np.random.RandomState(seed)


class FourRoomsOneGoalOptionsBase:
    
    def __init__(self, args):
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
        #         layout = """\
        # wwwwwwwwwwwww
        # w     w     w
        # w     w     w
        # w  g        w
        # w     w g   w
        # w     w     w
        # ww wwww     w
        # w     www www
        # w     w     w
        # w  g  w     w
        # w         g w
        # w     w     w
        # wwwwwwwwwwwww
        # """
        layout_G1 = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w  g  w
w           w
w     w     w
wwwwwwwwwwwww
"""
        layout_G2 = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     wwwgwww
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        layout_G3 = """\
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
w           w
w   g w     w
wwwwwwwwwwwww
"""
        option4 = """\
wwwwwwwwwwwww
w>>>>vw     w
w>>>>vw     w
w>>>>>      w
w>>>>^w     w
w>>>>^w     w
ww^wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        option5 = """\
wwwwwwwwwwwww
w>v<<<w     w
w>v<<<w     w
w>v<<<<     w
w>v<<<w     w
w>v<<<w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        option6 = """\
wwwwwwwwwwwww
w     wvvvvvw
w     wvvvvvw
w      <<<<<w
w     w^^^^^w
w     w^^^^^w
ww wwww^^^^^w
w     www^www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        option7 = """\
wwwwwwwwwwwww
w     w>>v<<w
w     w>>v<<w
w     >>>v<<w
w     w>>v<<w
w     w>>v<<w
ww wwww>>v<<w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        option8 = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
wwvwwww     w
wvvvvvwww www
wvvvvvw     w
wvvvvvw     w
w>>>>>      w
w^^^^^w     w
wwwwwwwwwwwww
"""
        option9 = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w>^<<<www www
w>^<<<w     w
w>^<<<w     w
w>^<<<<     w
w>^<<<w     w
wwwwwwwwwwwww
"""
        option10 = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w>>^<<w
w     w>^^^<w
w     >^^^^^w
w     w^^^^^w
wwwwwwwwwwwww
"""
        option11 = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     wwwvwww
w     wvv<<<w
w     wv<<<<w
w      <<<<<w
w     w^^^^^w
wwwwwwwwwwwww
"""
        #         layout = """\
        # wwwwwwwwwwwww
        # wgggggwgggggw
        # wgggggwgggggw
        # wgggggggggggw
        # wgggggwgggggw
        # wgggggwgggggw
        # wwwwwwwwwwwww
        # """
        if args is None:
            layout = layout_G1
            restart = "FixedRestart"
            self.stochasticity = False
        else:
            if "G1" in args:
                layout = layout_G1
            elif "G2" in args:
                layout = layout_G2
            elif "G3" in args:
                layout = layout_G3
            else:
                raise NotImplementedError
        
            if "FixedRestart" in args:
                restart = "FixedRestart"
            elif "RandomRestart" in args:
                restart = "RandomRestart"
            else:
                raise NotImplementedError
            
            if "DetDyna" in args:
                self.stochasticity = False
            elif "StochDyna" in args:
                self.stochasticity = True
            else:
                raise NotImplementedError

        self.occupancy = np.array([list(map(lambda c: 1 if c == 'w' else 0, line)) for line in layout.splitlines()])
        self.options = []
        for i in range(4, 12):
            self.options.append(np.array([list(map(
                lambda c: 0 if c == '^' else (
                    1 if c == 'v' else (
                        2 if c == '<' else (
                            3 if c == '>' else -1
                        )
                    )
                ), line)) for line in eval("option" + str(i)).splitlines()])
            )
        self.options = np.array(self.options)
        self.height, self.width = self.occupancy.shape
        self.goal_occupancy = np.array(
            [list(map(lambda c: 1 if c == 'g' else 0, line)) for line in layout.splitlines()]
        )
        
        # Four possible actions
        # 0: UP
        # 1: DOWN
        # 2: LEFT
        # 3: RIGHT
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
        self.total_statenum = statenum
        self.tocell = {v: k for k, v in self.tostate.items()}
        self.init_states = list(range(np.sum(self.occupancy == 0)))
        
        self.observation_space = Box(low=-100.0, high=100.0, shape=(self.total_statenum,), dtype=np.float32)
        self.tabular_dim = self.height * self.width
        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.reward_range = np.array([0, 1])
        self.metadata = None
        self.primitive_action_space = Discrete(4)
        
        if restart == 'FixedRestart':
            self.init_states = [self.tostate[(1, 1)]]
        elif restart == 'RandomRestart':
            for g in self.goals:
                self.init_states.remove(g)
        else:
            raise NotImplementedError
        
        self.eps_steps = 0
        self.last_reset_step = 0
    
    def render(self, show_goal=True):
        current_grid = np.array(self.occupancy)
        current_grid[self.current_cell[0], self.current_cell[1]] = 2
        if show_goal:
            goal_cell = self.tocell[self.goals[self.goal_idx]]
            current_grid[goal_cell[0], goal_cell[1]] = -1
        return current_grid
    
    def render_policy(self, optimal_actions):
        action_num_to_symbol = {0: '^', 1: 'v', 2: '<', 3: '>'}
        grid = np.array(self.occupancy).astype(str)
        for i in range(optimal_actions.shape[0]):
            grid[self.tocell[i]] = action_num_to_symbol[optimal_actions[i]]
        grid = np.array2string(grid, separator=',', formatter={'str_kind': lambda x: x})
        return grid
    
    def render_policy_over_options(self, optimal_options, only_action=False):
        action_num_to_symbol = {0: '^', 1: 'v', 2: '<', 3: '>'}
        grid = np.array(self.occupancy).astype(str)
        for i in range(optimal_options.shape[0]):
            if optimal_options[i] < 4:
                grid[self.tocell[i]] = action_num_to_symbol[optimal_options[i]]
            else:
                if only_action:
                    grid[self.tocell[i]] = action_num_to_symbol[self.option_action(self.tocell[i], optimal_options[i])[0]]
                else:
                    grid[self.tocell[i]] = optimal_options[i]
        grid = np.array2string(grid, separator=',', formatter={'str_kind': lambda x: x})
        return grid
    
    def render_policy_over_options_options_only(self, optimal_options, only_action=False):
        action_num_to_symbol = {0: '^', 1: 'v', 2: '<', 3: '>'}
        grid = np.array(self.occupancy).astype(str)
        for i in range(optimal_options.shape[0]):
            if only_action:
                grid[self.tocell[i]] = action_num_to_symbol[self.option_action(self.tocell[i], optimal_options[i])[0]]
            else:
                grid[self.tocell[i]] = optimal_options[i]
        grid = np.array2string(grid, separator=',', formatter={'str_kind': lambda x: x})
        return grid
    
    def reset(self):
        self.eps_steps = 0
        state = self.rng.choice(self.init_states)
        # state = self.rng.choice([0, 2, 4, 10, 14, 20, 24])
        self.goal_idx = self.rng.randint(self.goals.__len__())
        self.current_cell = self.tocell[state]
        obs = self.to_obs(self.current_cell)
        return obs
    
    def reset_test(self):
        self.eps_steps = 0
        # state = self.rng.choice(self.init_states)
        state = self.rng.choice([0, 2, 4, 10, 14, 20, 24])
        self.goal_idx = self.rng.randint(self.goals.__len__())
        self.current_cell = self.tocell[state]
        self.start_cell = self.current_cell
        obs = self.to_obs(self.current_cell)
        return obs
    
    def check_available_cells(self, cell):
        available_cells = []
        
        for action in range(self.primitive_action_space.n):
            next_cell = tuple(cell + self.directions[action])
            
            if not self.occupancy[next_cell]:
                available_cells.append(next_cell)
            else:
                available_cells.append(cell)
        
        return available_cells
    
    def option_action(self, cell, option):
        raise NotImplementedError

    def option_termination(self, cell, option):
        raise NotImplementedError

    def get_action_prob(self, cell, option, action):
        option_action, random_action = self.option_action(cell, option)
        if random_action:
            return 0.25
        if option_action == action:
            return 1.0
        else:
            return 0.0

    def get_all_options_action_probs(self, cell, action):
        options_probs = []
        for option in range(self.action_space.n):
            options_probs.append(self.get_action_prob(cell, option, action))
        return options_probs

    def get_termination_prob(self, cell, option):
        term = self.option_termination(cell, option)
        return term

    def get_all_options_term_probs(self, cell):
        options_probs = []
        for option in range(self.action_space.n):
            options_probs.append(self.get_termination_prob(cell, option))
        return options_probs
    
    def step(self, option):
        '''
		Takes a step in the environment with 2/3 probability. And takes a step in the
		other directions with probability 1/3 with all of them being equally likely.
		'''
        option, as_action = option
        option_term = None
        if as_action:
            action = option
            option_term = True
        else:
            action, _ = self.option_action(self.current_cell, option)

        assert (0 <= action and action <= 3)

        option_action_probs = self.get_all_options_action_probs(self.current_cell, action)
        
        if self.stochasticity:
            threshold = 5. / 9.
        else:
            threshold = 1.
        if np.random.uniform() < threshold:
            next_cell = tuple(self.current_cell + self.directions[action])

        else:
            dir = np.random.randint(self.primitive_action_space.n)
            next_cell = tuple(self.current_cell + self.directions[dir])

        if not self.occupancy[next_cell]:
        
            if self.tostate[next_cell] == self.goals[self.goal_idx]:
                reward = 1.0
                self.last_reset_step = self.eps_steps
                obs = self.reset()
            elif self.eps_steps - self.last_reset_step == 1000:
                self.last_reset_step = self.eps_steps
                reward = 0
                obs = self.reset()
            else:
                reward = 0
                self.current_cell = next_cell
                obs = self.to_obs(self.current_cell)
        else:
            reward = 0
            obs = self.to_obs(self.current_cell)

        if option_term is None:
            option_term = self.option_termination(self.current_cell, option)

        option_term_probs = self.get_all_options_term_probs(self.current_cell)
        info = {
            "option_term": option_term, "action": action,
            "options_action_probs": option_action_probs, "options_term_probs": option_term_probs
        }
        
        self.eps_steps += 1
        done = False
        
        return obs, reward, done, info
    
    def to_obs(self, cell):
        obs = np.zeros(self.total_statenum)
        obs[self.tostate[cell]] = 1
        return obs
    
    def seed(self, seed):
        # Random number generator
        self.rng = np.random.RandomState(seed)


class FourRoomsOneGoalOptionsandActions(FourRoomsOneGoalOptionsBase):
    def __init__(self, args):
        FourRoomsOneGoalOptionsBase.__init__(self, args)
        self.action_space = Discrete(12)
    
    def option_action(self, cell, option):
        if option <= 3:
            action = option
        else:
            action = self.options[option - 4, cell[0], cell[1]]
        
        random_action = False
        if action == -1:
            action = np.random.randint(0, 4)
            random_action = True
        return action, random_action
    
    def option_termination(self, cell, option):
        if option <= 3:
            term = True
        else:
            term = self.options[option - 4, cell[0], cell[1]] == -1
        return term


class FourRoomsOneGoalOptionsOnly(FourRoomsOneGoalOptionsBase):
    def __init__(self, args):
        FourRoomsOneGoalOptionsBase.__init__(self, args)
        self.action_space = Discrete(8)
    
    def option_action(self, cell, option):
        action = self.options[option, cell[0], cell[1]]

        random_action = False
        if action == -1:
            action = np.random.randint(0, 4)
            random_action = True
        return action, random_action
    
    def option_termination(self, cell, option):
        term = self.options[option, cell[0], cell[1]] == -1
        return term


class FourRoomsOneGoalActionsOnly(FourRoomsOneGoalOptionsBase):
    def __init__(self, args):
        FourRoomsOneGoalOptionsBase.__init__(self, args)
        self.action_space = Discrete(4)
    
    def option_action(self, cell, option):
        return option, False
    
    def option_termination(self, cell, option):
        return True


class CounterExamplefMax:
    
    def __init__(self, args):
#         layout = """\
# wwwwwwwwwww
# w        gw
# wwwwwwwwwww
# """
        layout = """\
wwwwwwwww
w   w   w
w w   w w
w   w   w
wwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c == 'w' else 0, line)) for line in layout.splitlines()])
        self.height, self.width = self.occupancy.shape
        self.goal_occupancy = np.array(
            [list(map(lambda c: 1 if c == 'g' else 0, line)) for line in layout.splitlines()])
        
        # Four possible actions
        # 0: UP
        # 1: DOWN
        # 2: LEFT
        # 3: RIGHT
        self.observation_space = Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
        self.tabular_dim = self.height * self.width
        # self.action_space = Discrete(4)
        # self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.action_space = Discrete(2)
        self.directions = [np.array((0, -1)), np.array((0, 1))]
        self.reward_range = np.array([-1, 10])
        self.tostate = {}
        self.goals = []
        self.terminals = []
        self.metadata = None
        
        statenum = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.goal_occupancy[i, j] == 1:
                    self.goals.append(statenum)
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1
        self.total_statenum = statenum
        self.tocell = {v: k for k, v in self.tostate.items()}
        
        self.init_states = list(range(np.sum(self.occupancy == 0)))
        for g in self.goals:
            self.init_states.remove(g)
        for t in self.terminals:
            self.init_states.remove(t)
        
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
        state = self.rng.choice(self.init_states)
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
        
        if self.current_cell == self.tocell[self.goals[0]]:
            next_cell = tuple(self.current_cell + np.array((0, -1)))
            reward = 10
        else:
            rnd_num = np.random.randint(100)
            if rnd_num < 10:
                next_cell = self.tocell[self.goals[0]]
                reward = 0
            else:
                next_cell = tuple(self.current_cell + self.directions[action])
                reward = 0
                if next_cell == self.tocell[self.goals[0]]:
                    next_cell = self.current_cell
                if next_cell == (1, 1):
                    reward = 1
        # elif self.current_cell == (1, 8) and action == 1:
        #     rnd_num = np.random.randint(100)
        #     if rnd_num < 10:
        #         next_cell = (1, 9)
        #         reward = 0
        #     else:
        #         next_cell = self.current_cell
        #         reward = 0
        # else:
        #     next_cell = tuple(self.current_cell + self.directions[action])
        #     reward = 0
        # print(self.current_cell, action, next_cell)
        if not self.occupancy[next_cell]:
            # if self.rng.uniform() < 1 / 3:
            #     available_cells = self.check_available_cells(self.current_cell)
            #     self.current_cell = available_cells[self.rng.randint(len(available_cells))]
            #
            # else:
            #     self.current_cell = next_cell
            self.current_cell = next_cell
        
        # state = self.tostate[self.current_cell]
        
        # When goal is reached, it is done
        # done = state == self.goals[self.goal_idx]
        # terminal = state == self.terminals[self.terminal_idx]
        # reward = done
        # reward = -0.1
        # if done:
        #     reward = 20
        # elif terminal:
        #     reward = 0
        # else:
        #     reward = -1
        obs = self.to_obs(self.current_cell, self.tocell[self.goals[self.goal_idx]])
        
        # info = {"goal": done}
        
        self.eps_steps += 1
        # if self.eps_steps == 100:
        #     done = True
        
        # done = done or terminal
        done = False
        info = {}
        
        return obs, reward, done, info
    
    def to_obs(self, cell, goal_cell):
        obs = np.zeros(self.total_statenum)
        obs[self.tostate[cell]] = 1
        return obs
    
    def seed(self, seed):
        # Random number generator
        self.rng = np.random.RandomState(seed)
    

class Corridol:
    
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
        #         layout = """\
        # wwwwwwwwwwwww
        # w     w     w
        # w     w     w
        # w  g        w
        # w     w g   w
        # w     w     w
        # ww wwww     w
        # w     www www
        # w     w     w
        # w  g  w     w
        # w         g w
        # w     w     w
        # wwwwwwwwwwwww
        # """
        layout = """\
wwwwwwwwwww
g         t
wwwwwwwwwww
"""
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
        self.goal_occupancy = np.array(
            [list(map(lambda c: 1 if c == 'g' else 0, line)) for line in layout.splitlines()])
        self.term_occupancy = np.array(
            [list(map(lambda c: 1 if c == 't' else 0, line)) for line in layout.splitlines()])
        
        # Four possible actions
        # 0: UP
        # 1: DOWN
        # 2: LEFT
        # 3: RIGHT
        self.observation_space = Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)
        self.tabular_dim = self.height * self.width
        # self.action_space = Discrete(4)
        # self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.action_space = Discrete(2)
        self.directions = [np.array((0, -1)), np.array((0, 1))]
        self.reward_range = np.array([-1, 10])
        self.tostate = {}
        self.goals = []
        self.terminals = []
        self.metadata = None
        
        statenum = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.goal_occupancy[i, j] == 1:
                    self.goals.append(statenum)
                if self.term_occupancy[i, j] == 1:
                    self.terminals.append(statenum)
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1
        self.total_statenum = statenum
        self.tocell = {v: k for k, v in self.tostate.items()}
        
        self.init_states = list(range(np.sum(self.occupancy == 0)))
        for g in self.goals:
            self.init_states.remove(g)
        for t in self.terminals:
            self.init_states.remove(t)
        
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
        self.terminal_idx = self.rng.randint(self.terminals.__len__())
        self.current_cell = self.tocell[state]
        obs = self.to_obs(self.current_cell, self.tocell[self.goals[self.goal_idx]])
        return obs
    
    def reset_test(self):
        self.eps_steps = 0
        # state = self.rng.choice(self.init_states)
        state = self.rng.choice(self.init_states)
        self.goal_idx = self.rng.randint(self.goals.__len__())
        # self.terminal_idx = self.rng.randint(self.terminals.__len__())
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
        terminal = state == self.terminals[self.terminal_idx]
        # reward = done
        # reward = -0.1
        if done:
            reward = 20
        elif terminal:
            reward = 0
        else:
            reward = -1
        obs = self.to_obs(self.current_cell, self.tocell[self.goals[self.goal_idx]])
        
        info = {"goal": done}
        
        self.eps_steps += 1
        if self.eps_steps == 100:
            done = True
            
        done = done or terminal
        
        return obs, reward, done, info
    
    def to_obs(self, cell, goal_cell):
        obs = np.zeros( self.total_statenum)
        obs[cell[1]] = 1
        return obs
    
    def seed(self, seed):
        # Random number generator
        self.rng = np.random.RandomState(seed)


class FourRoomsOneGoal:
    
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
#         layout = """\
# wwwwwwwwwwwww
# w     w     w
# w     w     w
# w  g        w
# w     w g   w
# w     w     w
# ww wwww     w
# w     www www
# w     w     w
# w  g  w     w
# w         g w
# w     w     w
# wwwwwwwwwwwww
# """
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
        self.goal_occupancy = np.array(
            [list(map(lambda c: 1 if c == 'g' else 0, line)) for line in layout.splitlines()])
        
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
        self.metadata = None
        
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
        # reward = -0.1
        reward = 1 * done
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


class FourRoomsFourGoals:
    
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
        self.goal_occupancy = np.array(
            [list(map(lambda c: 1 if c == 'g' else 0, line)) for line in layout.splitlines()])
        
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
        
        self.metadata = None
        
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


class FourRoomsSixteenGoals:
    
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
wg   gwg   gw
w     w     w
w           w
w     w     w
wg   gw     w
ww wwwwg   gw
wg   gwww www
w     wg   gw
w     w     w
w           w
wg   gwgg  gw
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
        self.goal_occupancy = np.array(
            [list(map(lambda c: 1 if c == 'g' else 0, line)) for line in layout.splitlines()])
        
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
        self.metadata = None
    
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


class FourRoomsAllGoals:
    
    def __init__(self):
        layout = """\
wwwwwwwwwwwww
wgggggwgggggw
wgggggwgggggw
wgggggggggggw
wgggggwgggggw
wgggggwgggggw
wwgwwwwgggggw
wgggggwwwgwww
wgggggwgggggw
wgggggwgggggw
wgggggggggggw
wgggggwgggggw
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
        self.goal_occupancy = np.array(
            [list(map(lambda c: 1 if c == 'g' else 0, line)) for line in layout.splitlines()])
        
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
        self.metadata = None
        
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
        self.metadata = None
    
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