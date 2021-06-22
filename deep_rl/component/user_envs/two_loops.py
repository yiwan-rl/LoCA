import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete


class TwoLoops:
	
	def __init__(self, args):
		self.observation_space = Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
		self.action_space = Discrete(2)
		self.directions = [0, 1]
		self.reward_range = np.array([-1, 10])
		self.tostate = {}
		self.goals = []
		self.terminals = []
		self.metadata = None
		self.size_left_loop = 5
		self.total_statenum = 9
		self.current_state = None
		self.eps_steps = 0
		self.args = args
	
	def reset(self):
		self.eps_steps = 0
		self.current_state = 0
		obs = self.to_obs(self.current_state)
		return obs
	
	def reset_test(self):
		self.eps_steps = 0
		self.current_state = 0
		obs = self.to_obs(self.current_state)
		return obs
	
	def step(self, action):
		if self.current_state == 9:
			reward = 100
			next_state = 0
		else:
			rnd_num = np.random.randint(100)
			if self.args == 1:
				if rnd_num < 0:
					next_state = 9
					reward = 0
				else:
					if self.current_state == 0:
						if action == 0:
							next_state = 1
						else:
							next_state = self.size_left_loop
					elif self.current_state < self.size_left_loop:
						next_state = (self.current_state + 1) % self.size_left_loop
					else:
						next_state = (self.current_state + 1) % self.total_statenum
					
					if next_state == 1:
						reward = 1
					elif self.current_state == 8:
						reward = 10
					else:
						reward = 0
			elif self.args == 2:
				if rnd_num < 2:
					next_state = 9
					reward = 0
				else:
					if self.current_state == 0:
						if action == 0:
							next_state = 1
						else:
							next_state = self.size_left_loop
					elif self.current_state < self.size_left_loop:
						next_state = (self.current_state + 1) % self.size_left_loop
					else:
						next_state = (self.current_state + 1) % self.total_statenum
					
					if next_state == 1:
						reward = 1
					elif self.current_state == 8:
						reward = 10
					else:
						reward = 0
			elif self.args == 3:
				if rnd_num < 0:
					next_state = 9
					reward = 0
				else:
					if self.current_state == 0:
						if action == 0:
							next_state = 1
						else:
							next_state = self.size_left_loop
					elif self.current_state < self.size_left_loop:
						next_state = (self.current_state + 1) % self.size_left_loop
					else:
						next_state = (self.current_state + 1) % self.total_statenum
					
					if next_state == 1:
						reward = 1
					elif self.current_state == 8:
						reward = 2
					else:
						reward = 0
			elif self.args == 4:
				if rnd_num < 2:
					next_state = 9
					reward = 0
				else:
					if self.current_state == 0:
						if action == 0:
							next_state = 1
						else:
							next_state = self.size_left_loop
					elif self.current_state < self.size_left_loop:
						next_state = (self.current_state + 1) % self.size_left_loop
					else:
						next_state = (self.current_state + 1) % self.total_statenum
					
					if next_state == 1:
						reward = 1
					elif self.current_state == 8:
						reward = 2
					else:
						reward = 0
			elif self.args == 5:
				if rnd_num < 0:
					next_state = 9
					reward = 0
				else:
					if self.current_state == 0:
						if action == 0:
							next_state = 1
						else:
							next_state = self.size_left_loop
					elif self.current_state < self.size_left_loop:
						next_state = (self.current_state + 1) % self.size_left_loop
					else:
						next_state = (self.current_state + 1) % self.total_statenum
					
					if next_state == 1:
						reward = 1
					elif next_state == 8:
						reward = 10
					else:
						reward = 0
			elif self.args == 6:
				if rnd_num < 2:
					next_state = 9
					reward = 0
				else:
					if self.current_state == 0:
						if action == 0:
							next_state = 1
						else:
							next_state = self.size_left_loop
					elif self.current_state < self.size_left_loop:
						next_state = (self.current_state + 1) % self.size_left_loop
					else:
						next_state = (self.current_state + 1) % self.total_statenum
					
					if next_state == 1:
						reward = 1
					elif next_state == 8:
						reward = 10
					else:
						reward = 0
			else:
				raise NotImplementedError
		obs = self.to_obs(next_state)
		# if self.current_state == 0:
		# 	print(self.current_state, action, next_state, reward)
		self.current_state = next_state
		
		info = {}
		
		self.eps_steps += 1
		
		done = False
		
		return obs, reward, done, info
	
	def to_obs(self, state):
		obs = np.zeros(self.total_statenum + 1)
		obs[state] = 1
		return obs
	
	def seed(self, seed):
		# Random number generator
		self.rng = np.random.RandomState(seed)