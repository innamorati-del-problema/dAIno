import os
import pygame
import random
from time import sleep

WIDTH = 800
HEIGHT = 400
FPS = 50

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

high_points = 0

# Assets
dino_run = [pygame.image.load(os.path.join("assets/Dino", "dino_run_1.png")),
            pygame.image.load(os.path.join("assets/Dino", "dino_run_2.png"))]

dino_duck = [pygame.image.load(os.path.join("assets/Dino", "dino_duck_1.png")),
             pygame.image.load(os.path.join("assets/Dino", "dino_duck_2.png"))]

dino_jump = [pygame.image.load(os.path.join("assets/Dino", "dino_still.png"))]

ptero = [pygame.image.load(os.path.join("assets/Ptero", "ptero_1.png")),
         pygame.image.load(os.path.join("assets/Ptero", "ptero_2.png"))]


cactus_small = [pygame.image.load(os.path.join("assets/Cactus", "cactus_small_1.png")),
                pygame.image.load(os.path.join("assets/Cactus", "cactus_small_2.png"))]

cactus_big = [pygame.image.load(os.path.join("assets/Cactus", "cactus_big_1.png")),
              pygame.image.load(os.path.join("assets/Cactus", "cactus_big_2.png"))]

background_img = pygame.image.load(os.path.join("assets", 'background.png'))

# Start settings
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("dAIno")
clock = pygame.time.Clock()


class Dinosaur:
    X_POS = 20
    Y_POS = 220
    Y_POS_DUCK = 237
    JUMP_VEL = 8

    def __init__(self):
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

    def update(self, userInput):
        if self.is_ducking:
            self.duck()
        if self.is_jumping:
            self.jump()
        if self.is_running:
            self.run()

        if self.animation_frame >= 20:
            self.animation_frame = 0

        # Keyboard bindings
        if userInput[pygame.K_SPACE] and not self.is_jumping:
            self.is_jumping = True
            self.is_running = False
            self.is_ducking = False
            self.jump_mul = 2.0

        elif userInput[pygame.K_DOWN] and not self.is_jumping:
            self.is_jumping = False
            self.is_running = False
            self.is_ducking = True

        elif userInput[pygame.K_UP] and not self.is_jumping:
            self.is_jumping = True
            self.is_running = False
            self.is_ducking = False
            self.jump_mul = 1.3

        elif not (self.is_jumping or userInput[pygame.K_DOWN]):
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

    def update(self, index):
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
        self.rect.y = 195
        self.animation_frame = 0

    def draw(self, screen):
        if self.animation_frame >= 19:
            self.animation_frame = 0
        screen.blit(self.image[self.animation_frame//10], self.rect)
        self.animation_frame += 1


def main():
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    running = True
    dino = Dinosaur()
    game_speed = 8

    x_pos_bg = 0
    y_pos_bg = 255

    obstacles = [None, None]

    points = 0

    font = pygame.font.Font('freesansbold.ttf', 20)

    def score():
        global game_speed, points, high_points
        points += 0.1

        # set game difficulty
        if points % 300 == 0 and game_speed < 20:
            game_speed += 1

        high_points = max(points, high_points)

        # show score
        text = font.render(f"Score: {str(int(points))}", True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (550, 40)
        screen.blit(text, textRect)

        # show highscore
        highscore = font.render(
            f"High Score: {str(int(high_points))}", True, (0, 0, 0))
        highscoreRect = highscore.get_rect()
        highscoreRect.center = (700, 40)
        screen.blit(highscore, highscoreRect)

    def background():
        global x_pos_bg, y_pos_bg, game_speed
        image_width = background_img.get_width()
        screen.blit(background_img, (x_pos_bg, y_pos_bg))
        screen.blit(background_img, (image_width + x_pos_bg, y_pos_bg))

        # repeat background
        if x_pos_bg <= -image_width:
            screen.blit(background_img, (image_width+x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)
        userInput = pygame.key.get_pressed()

        #run every seconds

        # choose obstacle
        if obstacles[0] == None:
            n = random.randint(0, 2)
            if n == 0:
                obstacles[0] = SmallCactus(cactus_small)
            if n == 1:
                obstacles[0] = LargeCactus(cactus_big)
            if n == 2:
                obstacles[0] = Ptero(ptero)

        if obstacles[1] == None and obstacles[0].rect.x < WIDTH//2:
            n = random.randint(0, 2)
            if n == 0:
                obstacles[1] = SmallCactus(cactus_small)
            if n == 1:
                obstacles[1] = LargeCactus(cactus_big)
            if n == 2:
                obstacles[1] = Ptero(ptero)

        for index, obstacle in enumerate(obstacles):
            if obstacle:
                obstacle.draw(screen)
                obstacle.update(index)
                if dino.dino_rect.colliderect(obstacle.rect):
                    # pygame.draw.rect(screen, RED, dino.dino_rect, 2)

                    sleep(0.5)
                    obstacles = []
                    running = False
                    main()

        background()
        score()

        dino.update(userInput)
        dino.draw(screen)
        pygame.display.update()


main()
