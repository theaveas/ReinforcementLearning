import sys
import random
from tarfile import BLOCKSIZE
import numpy as np
import pygame
import gym
from gym import spaces

BLOCK_SIZE = 10

class SnakeEnv_v2(gym.Env):
    """_summary_

    Args:
        gym (_type_): _description_
    """
    metadata = {'render.modes': ['human']}
    
    difficulty = 10
    SIZE = 360
    # action constants
    UP = 0
    DOWN = 1
    LEFT =  2
    RIGHT = 3
    
    def __init__(self):
        super(SnakeEnv_v2, self).__init__()
        pygame.init()
        pygame.display.set_caption('Snake Agent (Bob)')
        self.display = pygame.display.set_mode((self.SIZE, self.SIZE))
        self.clock = pygame.time.Clock()
        self.reset()
        
        # game variables
        self.head = [self.SIZE / 2, self.SIZE / 2]
        # flag bug
        self.body = [self.head, 
                    [self.head[0] - BLOCK_SIZE, self.SIZE / 2], 
                    [self.head[0] - (2 * BLOCK_SIZE), self.SIZE / 2]]
        self.food = [random.randrange(1, (self.SIZE // 10) * 10), random.randrange(1, (self.SIZE // 10)) * 10]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.counter = 0
        self.game_over = False
        
        # environment action and state space
        nums_of_actions = 4
        nums_of_obs = 4
        self.action_space = spaces.Discrete(nums_of_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=[nums_of_obs,], dtype=np.float32)
        
    def step(self, action):
        if action == self.UP: self.change_to = 'UP'
        elif action == self.DOWN: self.change_to = 'DOWN'
        elif action == self.LEFT: self.change_to = 'LEFT'
        elif action == self.RIGHT: self.change_to = 'RIGHT'
        # make sure agent can't turn 180 deg
        if self.change_to == 'UP' and self.direction != 'DOWN': 
            self.direction = 'UP'
        elif self.change_to == 'DOWN' and self.direction != 'UP': 
            self.direction = 'DOWN'
        elif self.change_to == 'LEFT' and self.direction != 'RIGHT': 
            self.direction = 'LEFT'
        elif self.change_to == 'RIGHT' and self.direction != 'LEFT': 
            self.direction = 'RIGHT'
        # moving the agent
        if self.direction == 'UP': self.head[1] -= 10
        elif self.direction == 'DOWN': self.head[1] += 10
        elif self.direction == 'LEFT': self.head[0] -= 10
        elif self.direction == 'RIGHT': self.head[0] += 10

        # growing mechanism
        self.body.insert(0, list(self.head))
        if self.head[0] == self.food[0] and self.head[1] == self.food[1]:
            self.counter = 0
            self.food_spawn = False
        else:
            self.body.pop()

        # spawning food randomly on the screen
        if not self.food_spawn:
            self.food = [random.randrange(1, (self.SIZE // 10)) * 10,
                         random.randrange(1, (self.SIZE // 10)) * 10 ]
        self.food_spawn = True
        
        # game over condition
        if self.head[0] > self.SIZE - 10 or self.head[0] < 0 or self.head[1] > self.SIZE - 10 or self.head[1] < 0 or self.counter > 100 * len(self.body):
            self.game_over = True
        # eatting itself
        for block in self.body[1:]:
            if self.head[0] == block[0] and self.head[1] == block[1]:
                self.game_over = True

        reward = 0
        done = self.game_over
        info = {}
        
        return np.array([self.head[0], self.head[1], self.food[0], self.food[1]], dtype=np.float32), reward, done, info
    
    def reset(self):
        # game variables
        self.head = [self.SIZE / 2, self.SIZE / 2]
        # flag bug
        self.body = [self.head, [self.head[0] - BLOCK_SIZE, self.SIZE / 2], [self.head[0] - (2 * BLOCK_SIZE), self.SIZE / 2]]
        self.food = [random.randint(0, (self.SIZE // 10)) * 10, random.randint(0, (self.SIZE // 10)) * 10 ]
        self.counter = 0
        self.food_spawn = True
        self.direction = 'Right'
        self.change_to = self.direction
        self.game_over = False
        return np.array([self.head[0], self.head[1], self.food[0], self.food[1]], dtype=np.float32)
        
    def render(self, mode='human'):
        self.display.fill(pygame.Color(255, 255, 255))
        # display snake body
        for pos in self.body:
            pygame.draw.rect(self.display, pygame.Color(0, 0, 0), pygame.Rect(pos[0], pos[1], 10, 10))
        # display snake food
        # refresh game screen
        pygame.display.update()
        # refresh rate
        self.clock.tick(self.difficulty)
        
    def close(self):
        sys.exit()
        
        
# food doesn't show up
# snake doesn't move