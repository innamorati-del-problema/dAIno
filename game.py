import os
import pygame
import random
from time import sleep
from enum import Enum


class Action(Enum):
    JUMP = 1
    SHORT_JUMP = 2
    DUCK = 3
    RUN = 4


pygame.init()
font = pygame.font.Font('arial.ttf', 25)

WIDTH = 800
HEIGHT = 400
FPS = 400

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class Daino:

    def __init__(self, width=WIDTH, heigth=HEIGHT):
        self.w = width
        self.h = heigth

        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("dAIno")
        self.clock = pygame.time.Clock()

        self.dino_run = [pygame.image.load(os.path.join("assets/Dino", "dino_run_1.png")),
                         pygame.image.load(os.path.join("assets/Dino", "dino_run_2.png"))]

        self.dino_duck = [pygame.image.load(os.path.join("assets/Dino", "dino_duck_1.png")),
                          pygame.image.load(os.path.join("assets/Dino", "dino_duck_2.png"))]

        self.dino_jump = [pygame.image.load(
            os.path.join("assets/Dino", "dino_still.png"))]

        self.ptero = [pygame.image.load(os.path.join("assets/Ptero", "ptero_1.png")),
                      pygame.image.load(os.path.join("assets/Ptero", "ptero_2.png"))]

        self.cactus_small = [pygame.image.load(os.path.join("assets/Cactus", "cactus_small_1.png")),
                             pygame.image.load(os.path.join("assets/Cactus", "cactus_small_2.png"))]

        self.cactus_big = [pygame.image.load(os.path.join("assets/Cactus", "cactus_big_1.png")),
                           pygame.image.load(os.path.join("assets/Cactus", "cactus_big_2.png"))]

        self.background_img = pygame.image.load(
            os.path.join("assets", 'background.png'))

        self.score = 0
        self.high_score = 0
        self.x_pos_bg = 0
        self.y_pos_bg = 255
        self.obstacles = [None, None]
        self.game_speed = 8
        self.dino = Dinosaur(self.dino_duck, self.dino_run, self.dino_run)

    def score_update(self):
        self.score += 0.01 * self.game_speed

        # set game difficulty
        if int(self.score) % 50 == 0 and self.game_speed < 20:
            self.game_speed += 1
            self.score += 1

        self.high_score = max(self.score, self.high_score)

        # show score
        text = font.render(f"Score: {str(int(self.score))}", True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (500, 40)
        self.screen.blit(text, textRect)

        # show highscore
        highscore = font.render(
            f"High Score: {str(int(self.high_score))}", True, (0, 0, 0))
        highscoreRect = highscore.get_rect()
        highscoreRect.center = (700, 40)
        self.screen.blit(highscore, highscoreRect)

    def background(self):
        image_width = self.background_img.get_width()
        self.screen.blit(self.background_img, (self.x_pos_bg, self.y_pos_bg))
        self.screen.blit(self.background_img,
                         (image_width + self.x_pos_bg, self.y_pos_bg))

        # repeat background
        if self.x_pos_bg <= -image_width:
            self.screen.blit(self.background_img,
                             (image_width+self.x_pos_bg, self.y_pos_bg))
            self.x_pos_bg = 0
        self.x_pos_bg -= self.game_speed

    def play_step(self, action):
        self.clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.screen.fill(WHITE)

        # choose obstacle
        if self.obstacles[0] == None:
            n = random.randint(0, 2)
            if n == 0:
                self.obstacles[0] = SmallCactus(self.cactus_small)
            if n == 1:
                self.obstacles[0] = LargeCactus(self.cactus_big)
            if n == 2:
                self.obstacles[0] = Ptero(self.ptero)

        if self.obstacles[1] == None and self.obstacles[0].rect.x < WIDTH//2:
            n = random.randint(0, 2)
            if n == 0:
                self.obstacles[1] = SmallCactus(self.cactus_small)
            if n == 1:
                self.obstacles[1] = LargeCactus(self.cactus_big)
            if n == 2:
                self.obstacles[1] = Ptero(self.ptero)

        # Check for game over
        reward = 0
        game_over = False
        for index, obstacle in enumerate(self.obstacles):
            if obstacle:
                obstacle.draw(self.screen)
                obstacle.update(index, self.obstacles, self.game_speed)
                if self.dino.dino_rect.colliderect(obstacle.rect):
                    reward = -100
                    game_over = True
                    return game_over, self.score, reward

        self.background()
        self.score_update()

        self.dino.update(action)
        self.dino.draw(self.screen)
        reward += 0.1 * self.game_speed
        pygame.display.update()

        return game_over, self.score, reward

    def reset(self):
        self.obstacles = [None, None]
        self.score = 0
        self.game_speed = 8


class Dinosaur:
    X_POS = 20
    Y_POS = 220
    Y_POS_DUCK = 237
    JUMP_VEL = 8

    def __init__(self, dino_duck, dino_run, dino_jump):
        self.duck_img = dino_duck
        self.run_img = dino_run
        self.jump_img = dino_jump
        self.jump_mul = 2.0

        self.is_running = True
        self.is_jumping = False
        self.is_ducking = False

        self.animation_frame = 0
        self.curr_jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = pygame.Rect(
            self.X_POS, self.Y_POS, self.image.get_width()-20,  self.image.get_height()-20)

    def update(self, ai_action):

        if self.is_ducking:
            self.duck()
        if self.is_jumping:
            self.jump()
        if self.is_running:
            self.run()

        if self.animation_frame >= 20:
            self.animation_frame = 0

        # actions
        # [1,0,0,0] -> Jump
        # [0,1,0,0] -> Duck
        # [0,0,1,0] -> Short Jump
        # [0,0,0,1] -> Run

        actions = [Action.JUMP, Action.DUCK, Action.SHORT_JUMP, Action.RUN]
        action = actions[ai_action.index(1)]

        # Keyboard bindings
        if action == Action.JUMP and not self.is_jumping:
            self.is_jumping = True
            self.is_running = False
            self.is_ducking = False
            self.jump_mul = 2.0

        elif action == Action.DUCK and not self.is_jumping:
            self.is_jumping = False
            self.is_running = False
            self.is_ducking = True

        elif action == Action.SHORT_JUMP and not self.is_jumping:
            self.is_jumping = True
            self.is_running = False
            self.is_ducking = False
            self.jump_mul = 1.3

        elif action == Action.RUN and not self.is_jumping:
            self.is_jumping = False
            self.is_running = True
            self.is_ducking = False

    def jump(self):
        self.image = self.jump_img[0]
        if self.curr_jump_vel < - self.JUMP_VEL:  # when dino touch ground after jump
            self.is_jumping = False
            self.curr_jump_vel = self.JUMP_VEL  # stop jump
            self.dino_rect.y = self.Y_POS
        if self.is_jumping:
            self.dino_rect.y -= self.curr_jump_vel * \
                self.jump_mul  # go up by jump velocity
            self.curr_jump_vel -= 1  # handle gravity

    def run(self):
        self.image = self.run_img[self.animation_frame // 10]
        self.dino_rect = pygame.Rect(
            self.X_POS, self.Y_POS, self.image.get_width()-20,  self.image.get_height()-20)
        self.animation_frame += 1

    def duck(self):
        self.image = self.duck_img[self.animation_frame // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.animation_frame += 1

    def draw(self, screen):
        # draw object on screen
        screen.blit(self.image, (self.dino_rect.x, self.dino_rect.y))


class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = WIDTH

    def update(self, index, obstacles, game_speed):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:  # if obstacle is out of screen it gets eliminated
            obstacles[index] = None

    def draw(self, screen):
        screen.blit(self.image[self.type], self.rect)


class SmallCactus(Obstacle):

    def __init__(self, image,):
        self.type = random.randint(0, 1)
        super().__init__(image, self.type)
        self.rect.y = 230


class LargeCactus(Obstacle):

    def __init__(self, image):
        self.type = random.randint(0, 1)
        super().__init__(image, self.type)
        self.rect.y = 220


class Ptero(Obstacle):

    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        # n = random.randint(0, 1)
        self.rect.y = 170  # if n == 0 else 170
        self.animation_frame = 0

    def draw(self, screen):
        if self.animation_frame >= 19:
            self.animation_frame = 0
        screen.blit(self.image[self.animation_frame//10], self.rect)
        self.animation_frame += 1
