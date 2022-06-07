from time import sleep
from game import Daino
from collections import deque
import numpy as np
from game import SmallCactus, LargeCactus, Ptero
import torch
import random
from model import Linear_QNet, QTrainer
import torch.nn as nn
import cv2

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

game = Daino()

n_games = 0
epsilon = 0  # randomness
gamma = 0.9  # discount rate
memory = deque(maxlen=MAX_MEMORY)
model = Linear_QNet(4, 1024, 4)
target_model = Linear_QNet(4, 1024, 4)
trainer = QTrainer(model, target_model,  lr=LR, gamma=gamma)
last_high_score = 0


def get_state(game):
    min_x = 800

    obs_width = 0
    obs_height = 0
    game_speed = 0

    ss1 = game.take_screeshot()
    
    game.play_step([0,0,0,0])
    ss2 = game.take_screeshot()

    mask1 = ss1.copy()[150:265, :, 1]
    mask2 = ss2[150:265, :, 1]

    mask1 = cv2.bitwise_not(mask1)
    mask2 = cv2.bitwise_not(mask2)                     
    

    cnts1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts1:
        (x,y,w,h) = cv2.boundingRect(cnt)
        ss1 = cv2.rectangle(ss1, (x, y+150), (x+w, y+h+150), (0,255,0), 1)
        cv2.imshow('frame', ss1)
        if x > 80 and x < min_x:
            obs_height = h+y
            obs_width = w
            min_x = x
    
    for cnt in cnts2:
        (x,y,w,h) = cv2.boundingRect(cnt)
        if x > 80 and x < min_x:
            game_speed = min_x - x



    state = [min_x//10, obs_height, obs_width, game_speed]

    print(state)
    return np.array(state, dtype=int)

    


def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


def train_long_memory():
    if len(memory) > BATCH_SIZE:
        mini_sample = random.sample(
            memory, BATCH_SIZE)  # List of tuples
    else:
        mini_sample = memory

    states, actions, rewards, next_states, dones = zip(*mini_sample)
    trainer.train_step(states, actions, rewards, next_states, dones)


def train_short_memory(state, action, reward, next_state, done):
    trainer.train_step(state, action, reward, next_state, done)


def get_action(state):
    # random moves: tradeoff exploration / exploitation
    epsilon = 4 - (n_games/100)
    final_move = [0, 0, 0, 0]
    if random.random() < epsilon:
        move = random.randint(0, 3)
        final_move[move] = 1
    else:
        state0 = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():

            prediction = model(state0)

            # pred_probab = nn.Softmax(dim=0)(prediction)
            move = prediction.argmax(0)
            final_move[move] = 1

    return final_move

not_saved=True
while True:
    if n_games % 50 == 0:
        trainer.target_model.load_state_dict(trainer.model.state_dict())

    if not_saved and game.score > 10000:
        model.save()
        not_saved = False

    
    state_old = get_state(game)

    if not game.dino.is_jumping:
        # get move
        final_move = get_action(state_old)
    else:
        game_over, score, reward = game.play_step([0, 0, 0, 1])

        if game_over:
            print(f"Game n: {n_games} Score: {int(score)} High score: {int(last_high_score)} {chr(13)}")
            # train long memory / experience replay, plot results
            game.reset()
            n_games += 1
            if random.randint(0, 100) < 300-n_games:
                game.game_speed = random.randint(9,20)
            
            if game.high_score > last_high_score:
                model.save()
            last_high_score = max(last_high_score, game.high_score)
            train_long_memory()

        continue

    # perform move and get new state
    game_over, score, reward = game.play_step(final_move)
    state_new = get_state(game)

    # train short memory
    # train_short_memory(state_old, final_move, reward, state_new, game_over)
    remember(state_old, final_move, reward, state_new, game_over)

    if game_over:
        # train long memory / experience replay, plot results
        print(f"Game n: {n_games} Score: {int(score)} High score: {int(last_high_score)} {chr(13)}")
        game.reset()
        n_games += 1
        if random.randint(0, 100) < 300-n_games:
                game.game_speed = random.randint(9,20)
        
        if game.high_score > last_high_score:
            model.save()
        last_high_score = max(last_high_score, game.high_score)
        train_long_memory()
