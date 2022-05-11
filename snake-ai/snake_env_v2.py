import sys
import random
from tarfile import BLOCKSIZE
import numpy as np
import pygame
import gym
from gym import spaces



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
        pygame.display.set_caption('Snake Agent (Bob)')
        self.display = pygame.display.set_mode((self.SIZE, self.SIZE))
        self.clock = pygame.time.Clock()
        self.reset()
        
        # game variables
        self.head = [self.SIZE / 2, self.SIZE / 2]
        self.prev_head = [self.SIZE / 2, self.SIZE / 2]
        # flag bug
        self.body = [[self.SIZE / 2, self.SIZE / 2], [self.SIZE / 2 - 10, self.SIZE / 2], [self.SIZE / 2 - (2 * 10), self.SIZE / 2]]
        self.food = [random.randrange(1, ((self.SIZE / 2) // 10) * 10),
                    random.randrange(1, ((self.SIZE / 2) // 10)) * 10]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.counter = 0
        self.game_over = False
        
        # environment action and state space
        num_of_actions = 4
        num_of_obs = 4
        self.action_space = spaces.Discrete(num_of_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_of_obs,), dtype=np.float32)
        
    def step(self, action):
        self.counter += 1
        self.move(action)
        # growing mechanism
        self.body.insert(0, list(self.head))
        if self.head[0] == self.food[0] and self.head[1] == self.food[1]:
            self.counter = 0
            self.food_spawn = False
            reward = 100
        else:
            self.body.pop()

        # spawning food randomly on the screen
        if not self.food_spawn:
            self.food = [random.randrange(1, (self.SIZE // 10)) * 10,
                         random.randrange(1, (self.SIZE // 10)) * 10]
        self.food_spawn = True
        
        # game over condition
        if self.is_collision():
            self.game_over = True
            reward = -100
        else:
            reward = 0
        
        # punnish if agent take too long to eat the food
        if self.counter > 50 * len(self.body):
            reward = -100
            self.game_over = True
                
        # reward mechanism
        reward = 0
        dist_head_to_food = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
        dist_prev_head_to_food = abs(self.prev_head[0] - self.food[0]) + abs(self.prev_head[1] - self.food[1])
        if dist_head_to_food > dist_prev_head_to_food:
            reward = -1
        elif dist_head_to_food < dist_prev_head_to_food:
            reward = 1
        else:
            reward = 0
        self.prev_head = self.head.copy()
        if self.game_over:
            reward = -100
        done = self.game_over
        info = {}
        print(dist_head_to_food, dist_prev_head_to_food)
        return np.array([self.head[0], self.head[1], self.food[0], self.food[1]], dtype=np.float32), reward, done, info
    
    def reset(self):
        # game variables
        self.head = [self.SIZE / 2, self.SIZE / 2]
        self.prev_head = [self.SIZE / 2, self.SIZE / 2]
        self.body = [[self.SIZE / 2, self.SIZE / 2], [self.SIZE / 2 - 10, self.SIZE / 2], [self.SIZE / 2 - (2 * 10), self.SIZE / 2]]
        self.food = [random.randrange(1, (self.SIZE // 10)) * 10,
                     random.randrange(1, (self.SIZE // 10)) * 10 ]
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
        pygame.draw.rect(self.display, pygame.Color(255, 0, 0), pygame.Rect(self.food[0], self.food[1], 10, 10))
        # refresh game screen
        pygame.display.update()
        # refresh rate
        self.clock.tick(self.difficulty)
        
    def close(self):
        pygame.quit()
        sys.exit()
      
    # snake function 
    def is_collision(self):
        """Check if the snake hit itself or hit the wall
        """
        if self.head[0] > self.SIZE - 10 or self.head[0] < 0:
            return True
        if self.head[1] > self.SIZE - 10 or self.head[1] < 0:
            return True
        # eatting itself
        for block in self.body[1:]:
            if self.head[0] == block[0] and self.head[1] == block[1]:
                return True
        return False
            
    def move(self, action):
        """Move the snake during the game

        Args:
            action (int): Action for the agent
        """
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
        if self.direction == 'UP':
            self.head[1] -= 10
        elif self.direction == 'DOWN':
            self.head[1] += 10
        elif self.direction == 'LEFT':
            self.head[0] -= 10
        elif self.direction == 'RIGHT':
            self.head[0] += 10
