import pygame
import random
import numpy as np
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# ⚙️ Налаштування (Settings)
# ==========================================
WIDTH, HEIGHT = 400, 600
FPS = 60
SHIP_WIDTH, SHIP_HEIGHT = 40, 40
ASTEROID_SIZE = 40
ASTEROID_SPEED = 7
MAX_ASTEROIDS = 3

# Параметри дискретизації
SHIP_BINS, AST_X_BINS, AST_Y_BINS = 10, 10, 15

# Гіперпараметри RL
TRAIN_EPISODES = 3000
LEARNING_RATE, DISCOUNT_RATE = 0.1, 0.99
EPSILON_START, EPSILON_DECAY, EPSILON_MIN = 1.0, 0.0005, 0.01

# ==========================================
# ⚡ Клас Середовища (Environment)
# ==========================================
class SpaceDodgerEnv:
    def __init__(self):
        self.action_space = 3 
        self.reset()
    
    def reset(self):
        self.ship_x = WIDTH // 2
        self.ship_y = HEIGHT - SHIP_HEIGHT - 20
        self.asteroids = [[random.randint(0, WIDTH-ASTEROID_SIZE), -random.randint(50, 600)] for _ in range(MAX_ASTEROIDS)]
        return self._get_state()
    
    def _get_state(self):
        upcoming = [a for a in self.asteroids if a[1] < self.ship_y + SHIP_HEIGHT]
        closest = min(upcoming, key=lambda a: abs(self.ship_y - a[1])) if upcoming else self.asteroids[0]
        ship_bin = int(self.ship_x / WIDTH * (SHIP_BINS - 1))
        ast_x_bin = int(closest[0] / WIDTH * (AST_X_BINS - 1))
        ast_y_bin = int(max(0, min(closest[1], HEIGHT)) / HEIGHT * (AST_Y_BINS - 1))
        return (ship_bin * AST_X_BINS * AST_Y_BINS) + (ast_x_bin * AST_Y_BINS) + ast_y_bin
    
    def step(self, action):
        speed = 15
        if action == 0:   self.ship_x = max(0, self.ship_x - speed)
        elif action == 2: self.ship_x = min(WIDTH - SHIP_WIDTH, self.ship_x + speed)
        
        done, reward = False, 0.1
        for a in self.asteroids:
            a[1] += ASTEROID_SPEED
            if a[1] > HEIGHT:
                a[0], a[1] = random.randint(0, WIDTH-ASTEROID_SIZE), -ASTEROID_SIZE
                reward += 2.0
        
        ship_rect = pygame.Rect(self.ship_x, self.ship_y, SHIP_WIDTH, SHIP_HEIGHT)
        for a in self.asteroids:
            if ship_rect.colliderect(pygame.Rect(a[0], a[1], ASTEROID_SIZE, ASTEROID_SIZE)):
                reward, done = -15, True
        return self._get_state(), reward, done

# ==========================================
# 🎮 Режим гри (Ручний або ШІ)
# ==========================================
def run_game(env, qtable=None, manual=False):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Space Dodger: " + ("РУЧНЕ КЕРУВАННЯ" if manual else "ШТУЧНИЙ ІНТЕЛЕКТ"))
    clock = pygame.time.Clock()
    
    try:
        ship_img = pygame.transform.scale(pygame.image.load("ship.png").convert_alpha(), (SHIP_WIDTH, SHIP_HEIGHT))
        ast_img = pygame.transform.scale(pygame.image.load("asteroid.png").convert_alpha(), (ASTEROID_SIZE, ASTEROID_SIZE))
    except: 
        ship_img = ast_img = None

    state = env.reset()
    running = True
    while running:
        action = 1 
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                pygame.quit() # Важливо коректно закрити pygame
                return False

        if manual:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]: action = 0
            elif keys[pygame.K_RIGHT]: action = 2
        else:
            action = np.argmax(qtable[state])

        # Тут була помилка: замість self використовуємо env
        state, _, done = env.step(action)

        screen.fill((10, 10, 30))
        
        # Малюємо корабель, звертаючись до env
        if ship_img: 
            screen.blit(ship_img, (env.ship_x, env.ship_y))
        else: 
            pygame.draw.rect(screen, (0, 255, 0), (env.ship_x, env.ship_y, SHIP_WIDTH, SHIP_HEIGHT))
        
        # Малюємо астероїди, звертаючись до env
        for a in env.asteroids:
            if ast_img: 
                screen.blit(ast_img, (a[0], a[1]))
            else: 
                pygame.draw.rect(screen, (255, 50, 50), (a[0], a[1], ASTEROID_SIZE, ASTEROID_SIZE))

        pygame.display.flip()
        clock.tick(FPS)
        
        if done:
            if manual: print("💥 Бум! Спробуйте ще раз.")
            state = env.reset()
            
    pygame.quit()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Space Dodger: " + ("РУЧНЕ КЕРУВАННЯ" if manual else "ШТУЧНИЙ ІНТЕЛЕКТ"))
    clock = pygame.time.Clock()
    
    try:
        ship_img = pygame.transform.scale(pygame.image.load("ship.png").convert_alpha(), (SHIP_WIDTH, SHIP_HEIGHT))
        ast_img = pygame.transform.scale(pygame.image.load("asteroid.png").convert_alpha(), (ASTEROID_SIZE, ASTEROID_SIZE))
    except: ship_img = ast_img = None

    state = env.reset()
    running = True
    while running:
        action = 1 # За замовчуванням стоїмо
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False

        if manual:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]: action = 0
            elif keys[pygame.K_RIGHT]: action = 2
        else:
            action = np.argmax(qtable[state])

        state, _, done = env.step(action)

        screen.fill((10, 10, 30))
        if ship_img: screen.blit(ship_img, (env.ship_x, env.ship_y))
        else: pygame.draw.rect(screen, (0, 255, 0), (env.ship_x, env.ship_y, SHIP_WIDTH, SHIP_HEIGHT))
        for a in self.asteroids:
            if ast_img: screen.blit(ast_img, (a[0], a[1]))
            else: pygame.draw.rect(screen, (255, 50, 50), (a[0], a[1], ASTEROID_SIZE, ASTEROID_SIZE))

        pygame.display.flip()
        clock.tick(FPS)
        if done:
            if manual: print("💥 Бум! Спробуйте ще раз."); state = env.reset()
            else: state = env.reset()
    pygame.quit()

# ==========================================
# 🧠 Навчання та Збереження
# ==========================================
def train_agent(env):
    state_size = SHIP_BINS * AST_X_BINS * AST_Y_BINS
    qtable = np.zeros((state_size, env.action_space))
    epsilon, history = EPSILON_START, []

    print(f"🚀 ШІ починає вчитися ({TRAIN_EPISODES} епізодів)...")
    for _ in tqdm(range(TRAIN_EPISODES)):
        state, total_reward, done = env.reset(), 0, False
        while not done:
            action = random.randint(0, 2) if random.random() < epsilon else np.argmax(qtable[state])
            next_state, reward, done = env.step(action)
            qtable[state, action] += LEARNING_RATE * (reward + DISCOUNT_RATE * np.max(qtable[next_state]) - qtable[state, action])
            state, total_reward = next_state, total_reward + reward
        epsilon = max(EPSILON_MIN, epsilon - EPSILON_DECAY)
        history.append(total_reward)
    
    os.makedirs("space_dodger_rl", exist_ok=True)
    np.save("space_dodger_rl/q_table.npy", qtable)
    return qtable

if __name__ == "__main__":
    env = SpaceDodgerEnv()
    
    print("🎮 КРОК 1: Спробуй керувати сам (Стрілки ВЛІВО/ВПРАВО).")
    print("Закрий вікно гри, щоб перейти до навчання ШІ.")
    run_game(env, manual=True)

    path = "space_dodger_rl/q_table.npy"
    qtable = np.load(path) if os.path.exists(path) else None

    if qtable is None or input("\n🧠 Модель знайдена. Перенавчити? (y/n): ").lower() == 'y':
        qtable = train_agent(env)

    print("\n🏆 КРОК 2: Дивись, як грає ШІ!")
    run_game(env, qtable=qtable, manual=False)