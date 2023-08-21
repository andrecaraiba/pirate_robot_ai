import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple


pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

BLOCK_SIZE = 100
GRID_SIZE = 7
SPEED = 15

class RobotGame:
    
    def __init__(self, w=700, h=700):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        #init game state
        self.direction = Direction.RIGHT
        
        #initial_x = (GRID_SIZE // 2) * BLOCK_SIZE
        #initial_y = (GRID_SIZE // 2) * BLOCK_SIZE
        
        initial_x = 0
        initial_y = 0
        self.head = Point(initial_x, initial_y)
        
        self.snake = [self.head]
    
        self.score = 0
        self.food = None
        self.treasure = None
        self._place_food()
        self._place_treasure()
        self.frame_iteration = 0
        

    def _place_food(self):
        while True:
            x = random.randint(0, GRID_SIZE - 1) * BLOCK_SIZE
            y = random.randint(0, GRID_SIZE - 1) * BLOCK_SIZE
            self.food = Point(x, y)
            
            if self.food not in self.snake:
                break
    
    def _place_treasure(self):
        while True:
            x = 6 * BLOCK_SIZE
            y = 6 * BLOCK_SIZE
            self.treasure = Point(x, y)
            
            if self.treasure not in self.snake:
                break
        
    def play_step(self, action):
        self.frame_iteration += 1
        self.score = self.score - 1
        reward = self.score
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            #if event.type == pygame.KEYDOWN:
             #   if event.key == pygame.K_LEFT:
               #     action = [0, 1, 0, 0]
              #  elif event.key == pygame.K_RIGHT:
                #    action = [1, 0, 0, 0]
               # elif event.key == pygame.K_UP:
                #    action = [0, 0, 0, 1]
               # elif event.key == pygame.K_DOWN:
                #    action = [0, 0, 1, 0]
        
        # 2. move
        self._move(action)
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        #reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100:
            self.score -= 16
            game_over = True
            reward -= 16
            return reward, game_over, self.score + 100

        # 4. place new obstacle or just move    
        if self.head == self.food:
            self.score -= 20
            game_over = True
            reward -= -20
            self._place_food()
            return reward, game_over, self.score + 100

        if self.head == self.treasure:
            print('Treasure found!')
            self.score += 20
            reward += 20
            game_over = True
            return reward, game_over, self.score + 100
        
        self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if (
            pt.x >= self.w or pt.x < 0 or
            pt.y >= self.h or pt.y < 0 or
            
            self.head in self.snake[1:] #remove
        ):
            return True
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.treasure.x, self.treasure.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        # [1, 0, 0, 0] -> right
        # [0, 1, 0, 0] -> left
        # [0, 0, 1, 0] -> down
        # [0, 0, 0, 1] -> up


        if np.array_equal(action, [1, 0, 0, 0]):
            self.direction = Direction.RIGHT
        
        elif np.array_equal(action, [0, 1, 0, 0]):
            self.direction = Direction.LEFT
            
        elif np.array_equal(action, [0, 0, 1, 0]):
            self.direction = Direction.DOWN
        else:
            self.direction = Direction.UP
            


        x = self.head.x
        y = self.head.y
        
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            
"""
if __name__ == '__main__':
    game = RobotGame(w=BLOCK_SIZE * GRID_SIZE, h=BLOCK_SIZE * GRID_SIZE)
    
    while True:
        # [1, 0, 0, 0] -> right
        # [0, 1, 0, 0] -> left
        # [0, 0, 1, 0] -> down
        # [0, 0, 0, 1] -> up
        final_move = [0, 0, 0, 0]    
        move = random.randint(0, 2)
        final_move[move] = 1
        game_over, score = game.play_step(final_move)
        
        if game_over:
            break
        
    print('Final Score:', score)
    pygame.quit()
    """