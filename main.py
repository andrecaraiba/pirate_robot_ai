import torch
from torch.distributions import Categorical
import random
import numpy as np
from collections import deque
from environment import RobotGame, Direction, Point
from model import Linear_QNet, QTrainer, PolicyNet, PolicyTrainer
from helper import plot

BLOCK_SIZE = 100
GRID_SIZE = 7

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001





class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.store_ep = deque(maxlen=MAX_MEMORY)
        self.model = PolicyNet(16, 256, 4)
        self.trainer = PolicyTrainer(self.model, lr=LR, gamma=self.gamma)
        #self.model = Linear_QNet(16, 256, 4)
        #self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


        
    def get_state(self, game):
        '''
        [[lu, u, lr],
         [l, pt, r],
         [ld, d, rd]]
        '''
        pt = game.head
        point_l = Point(pt.x - BLOCK_SIZE, pt.y)
        point_r = Point(pt.x + BLOCK_SIZE, pt.y)
        point_u = Point(pt.x, pt.y - BLOCK_SIZE)
        point_d = Point(pt.x, pt.y + BLOCK_SIZE)
        #point_lu = Point(pt.x - BLOCK_SIZE, pt.y - BLOCK_SIZE)
        #point_ld = Point(pt.x - BLOCK_SIZE, pt.y + BLOCK_SIZE)
        #point_ru = Point(pt.x + BLOCK_SIZE, pt.y - BLOCK_SIZE)
        #point_rd = Point(pt.x + BLOCK_SIZE, pt.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (game.is_collision(point_r)),  # Danger right
            (game.is_collision(point_l)),  # Danger left
            (game.is_collision(point_u)),  # Danger up
            (game.is_collision(point_d)),  # Danger down

            # Danger diagonal
            #(game.is_collision(point_ru)),  # Danger right up
            #(game.is_collision(point_rd)),  # Danger right down
            #(game.is_collision(point_lu)),  # Danger left up
            #(game.is_collision(point_ld)),  # Danger left down

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            # food left
            ((pt.x == game.food.x + BLOCK_SIZE) and (pt.y == game.food.y)) or
            ((pt.x == game.food.x + BLOCK_SIZE) and (pt.y == game.food.y + BLOCK_SIZE)) or
            ((pt.x == game.food.x + BLOCK_SIZE) and (pt.y == game.food.y - BLOCK_SIZE)),

            # food right
            ((pt.x == game.food.x - BLOCK_SIZE) and (pt.y == game.food.y)) or
            ((pt.x == game.food.x - BLOCK_SIZE) and (pt.y == game.food.y + BLOCK_SIZE)) or
            ((pt.x == game.food.x - BLOCK_SIZE) and (pt.y == game.food.y - BLOCK_SIZE)),

            # food up
            ((pt.y == game.food.y + BLOCK_SIZE) and (pt.x == game.food.x)) or
            ((pt.y == game.food.y + BLOCK_SIZE) and (pt.x == game.food.x + BLOCK_SIZE)) or
            ((pt.y == game.food.y + BLOCK_SIZE) and (pt.x == game.food.x - BLOCK_SIZE)),
           
           # food down
            ((pt.y == game.food.y - BLOCK_SIZE) and (pt.x == game.food.x)) or
            ((pt.y == game.food.y - BLOCK_SIZE) and (pt.x == game.food.x + BLOCK_SIZE)) or
            ((pt.y == game.food.y - BLOCK_SIZE) and (pt.x == game.food.x - BLOCK_SIZE)),

            

            # Treasure location
            # treasure left
            ((pt.x == game.treasure.x + BLOCK_SIZE) and (pt.y == game.treasure.y)) or
            ((pt.x == game.treasure.x + BLOCK_SIZE) and (pt.y == game.treasure.y + BLOCK_SIZE)) or
            ((pt.x == game.treasure.x + BLOCK_SIZE) and (pt.y == game.treasure.y - BLOCK_SIZE)),

            # treasure right
            ((pt.x == game.treasure.x - BLOCK_SIZE) and (pt.y == game.treasure.y)) or
            ((pt.x == game.treasure.x - BLOCK_SIZE) and (pt.y == game.treasure.y + BLOCK_SIZE)) or
            ((pt.x == game.treasure.x - BLOCK_SIZE) and (pt.y == game.treasure.y - BLOCK_SIZE)),

            # treasure up
            ((pt.y == game.treasure.y + BLOCK_SIZE) and (pt.x == game.treasure.x)) or
            ((pt.y == game.treasure.y + BLOCK_SIZE) and (pt.x == game.treasure.x + BLOCK_SIZE)) or
            ((pt.y == game.treasure.y + BLOCK_SIZE) and (pt.x == game.treasure.x - BLOCK_SIZE)),

            # treasure down
            ((pt.y == game.treasure.y - BLOCK_SIZE) and (pt.x == game.treasure.x)) or
            ((pt.y == game.treasure.y - BLOCK_SIZE) and (pt.x == game.treasure.x + BLOCK_SIZE)) or
            ((pt.y == game.treasure.y - BLOCK_SIZE) and (pt.x == game.treasure.x - BLOCK_SIZE)),
        ]

        return np.array(state, dtype=int)
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def store_episode(self, state, action, reward):
        self.store_ep.append((state, action, reward))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def train_policy(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.store_ep, BATCH_SIZE)
        else:
            mini_sample = self.store_ep

        states, actions, rewards = zip(*mini_sample)
        print("rewards: ", rewards)
        self.trainer.train_step(states, actions, rewards)

    def get_action(self, state): # mudar para que ele possa agir de acordo com a policy
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def get_action_police(self, state):
        final_move = [0, 0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        state0 = torch.unsqueeze(state0, 0)
        prediction = self.model(state0)
        m = Categorical(prediction)
        move = m.sample()
        final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = RobotGame(w=BLOCK_SIZE * GRID_SIZE, h=BLOCK_SIZE * GRID_SIZE)
    agent = Agent()
    while True:
        # [1, 0, 0, 0] -> right
        # [0, 1, 0, 0] -> left
        # [0, 0, 1, 0] -> down
        # [0, 0, 0, 1] -> up

        #get old state
        state_old = agent.get_state(game)

        #get move
        #final_move = agent.get_action(state_old)
        final_move = agent.get_action_police(state_old)   

        reward, game_over, score = game.play_step(final_move)

        #store episode
        agent.store_episode(state_old, final_move, reward) 

        #state_new = agent.get_state(game)  #remove

        #train short memory
        #agent.train_short_memory(state_old, final_move, reward, state_new, game_over) #remove
                
        #remember
        #agent.remember(state_old, final_move, reward, state_new, game_over)  #remove
        
        if game_over:
            #train long memory, plot result
            agent.n_games += 1
            agent.train_policy()
            #agent.train_long_memory() #remove
            game.reset()
        
            if score > record:
                record = score
                agent.model.save()
                #agent.model.save()
            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            

if __name__ == '__main__':
    train()
