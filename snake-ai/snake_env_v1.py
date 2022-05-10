# import module for snake agent
import sys
import random
from collections import namedtuple
from tarfile import BLOCKSIZE
import numpy as np
import gym
import pygame
from gym import spaces


Point = namedtuple('Point', 'x, y')

class SnakeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    difficulty = 10
    # window size
    HEIGHT = 360
    WIDTH = 360
    BLOCKSIZE = 10
    # action const
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    def __init__(self):
        super(SnakeEnv, self).__init__()
        pygame.init()
        # initialize game window
        pygame.display.set_caption('Snake Agent')
        self.window = pygame.display.set_mode((self.HEIGHT, self.WIDTH))
        # fps controller
        self.fps_controller = pygame.time.Clock()
        # game variable
        self.snake_head = Point(self.HEIGHT / 2, self.WIDTH / 2)
        self.snake_body = [self.snake_head,
                           Point(self.snake_head.x - self.BLOCKSIZE, self.snake_head.y), 
                           Point(self.snake_head.x - (2 * self.BLOCKSIZE), self.snake_head.y)]
        self.snake_prev_head = Point(self.HEIGHT / 2, self.WIDTH / 2)
        self.counter = 0
        # self.score = 0
        self.direction = 'Right'
        self.change_to = self.direction
        self.over = False
        self.food = None
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        num_of_actions = 4
        num_of_states = 4
        self.action_space = spaces.Discrete(num_of_actions)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(num_of_states,), dtype=np.float32)
    def step(self, action):
        # discount factor make the snake eat the food asap
        self.counter += 1
        # return a negative reward after 100 step without eating the food
        if self.counter > 100 * len(self.snake_body):
            return np.array([self.snake_head.x, self.snake_head.y,
                            self.food.x, self.food.y], dtype=np.float32), -100, True, {}
        # moving the snake
        self._move(action)
        # growing mechanism
        self.snake_body.insert(0, self.snake_head)
        # spawning food on the window
        if self.snake_head == self.food:
            self._place_food()
            reward = 100
            self.counter = 0
        else:
            self.snake_body.pop()
            # add score and reward
        # check if the snake collide with the wall or itself
        self.over = self.is_collision()
        # reward machanism
        reward = 0
        # snake agent go out of the window
        if self.over:
            reward = -100
        elif abs(self.snake_head.x - self.food.x) + abs(self.snake_head.y - self.food.y) > abs(self.snake_prev_head.x- self.food.x) + abs(self.snake_prev_head.y - self.food.y):
            reward = -3
        elif abs(self.snake_head.x - self.food.x) + abs(self.snake_head.y - self.food.y) < abs(self.snake_prev_head.x- self.food.x) + abs(self.snake_prev_head.y - self.food.y):
            reward = 1
        self.snake_prev_head = self.snake_head
        done = self.over
        info = {}
        return np.array([self.snake_head.x, self.snake_head.y,
                self.food.x, self.food.y], dtype=np.float32), reward, done, info
    def reset(self):
        self.snake_head = Point(self.HEIGHT / 2, self.WIDTH / 2)
        self.snake_body = [self.snake_head,
                           Point(self.snake_head.x - self.BLOCKSIZE, self.snake_head.y), 
                           Point(self.snake_head.x - (2 * self.BLOCKSIZE), self.snake_head.y)]
        self.snake_prev_head = Point(self.HEIGHT / 2, self.WIDTH / 2)
        self.counter = 0
        self.direction = 'Right'
        self.change_to = self.direction
        self.over = False
        self.food = None
        self._place_food()
        # return only obs *reward, done, info can't be included
        return np.array([self.snake_head.x, self.snake_head.y,
                self.food.x, self.food.y], dtype=np.float32) 
    def render(self, mode='human'):
        self.window.fill(pygame.Color(255, 255, 255))
        for pos in self.snake_body:
            pygame.draw.rect(self.window, pygame.Color(0, 0, 0), 
                             pygame.Rect(pos.x, pos.y, self.BLOCKSIZE, self.BLOCKSIZE))
        # snake game
        pygame.draw.rect(self.window, pygame.Color(255, 0, 0), pygame.Rect(self.food.x, self.food.y, self.BLOCKSIZE, self.BLOCKSIZE))
        # self.show_score(1, self.white, 'consolas', 20)
        # refresh game screen
        pygame.display.update()
        # game difficulty
        self.fps_controller.tick(self.difficulty)
    def close (self):
        pygame.quit()
        sys.exit()
    def _move(self, action):
        if action == self.UP:
            self.change_to = 'UP'
        if action == self.DOWN:
            self.change_to = 'DOWN'
        if action == self.LEFT:
            self.change_to = 'LEFT'
        if action == self.RIGHT:
            self.change_to = 'RIGHT'
        # make sure that the snake don't turn 180 deg
        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'
        x = self.snake_head.x
        y = self.snake_head.y
        if self.direction == 'UP':
            x -= self.BLOCKSIZE
        if self.direction == 'DOWN':
            x += self.BLOCKSIZE
        if self.direction == 'LEFT':
            y -= self.BLOCKSIZE
        if self.direction == 'RIGHT':
            y += self.BLOCKSIZE
        self.snake_head = Point(x, y)
    def _place_food(self):
        x = random.randint(0, self.HEIGHT // 10) * 10 
        y = random.randint(0, self.WIDTH // 10) * 10
        self.food = Point(x, y)
        if self.food in self.snake_body:
            self._place_food()
    # game over
    def is_collision(self):
        if self.snake_head.x > self.HEIGHT - self.BLOCKSIZE or self.snake_head.x < 0:
            return True
        if self.snake_head.y > self.WIDTH - self.BLOCKSIZE or self.snake_head.y < 0:
            return True
        # eating self
        for self.snake_head in self.snake_body[1:]:
            return True
        # otherwise
        return False
    def show_score(self, choice, color, font, size):
        font = pygame.font.SysFont(font, size)
        score_srf = font.render('Score:' + str(self.score), True, color)
        score_rect = score_srf.get_rect()
        if choice == 1:
            score_rect.midtop(self.HEIGHT / 10, 15)
        else:
            score_rect.midtop(self.HEIGHT / 2, self.WIDTH/1.25)         
