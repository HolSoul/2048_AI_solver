import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from game_env import Game2048Env  # Наша новая среда
from ai_solver import DQNAgent, DEVICE  # Наш агент (без изменений)

# --- Параметры обучения ---
NUM_EPISODES = 3000
MAX_STEPS_PER_EPISODE = 2000 # Ограничим, чтобы избежать бесконечных циклов

# --- Инициализация ---
env = Game2048Env()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size, 
                 replay_buffer_size=20000, 
                 batch_size=128,          
                 learning_rate=0.00025,
                 gamma=0.99,
                 epsilon_decay=0.99999)

# --- Хранение результатов ---
scores_history = deque(maxlen=100)
avg_scores_history = []
max_tile_history = deque(maxlen=100)

print(f"Начинаем обучение на {DEVICE}...")
best_avg_score = -float('inf')

# --- Цикл обучения ---
for episode in range(1, NUM_EPISODES + 1):
    state, info = env.reset()
    total_reward_this_episode = 0
    
    for step in range(MAX_STEPS_PER_EPISODE):
        action = agent.act(state)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        total_reward_this_episode += reward
        
        if len(agent.memory) > agent.batch_size:
            agent.replay()

        if done:
            break
            
    scores_history.append(info['score'])
    max_tile_history.append(info['max_tile'])
    
    avg_score = np.mean(scores_history)
    avg_max_tile = np.mean(max_tile_history)
    avg_scores_history.append(avg_score)

    print(f"Эпизод: {episode}/{NUM_EPISODES}, Счет: {info['score']}, "
          f"Макс.плитка: {info['max_tile']}, Сред.счет(100): {avg_score:.2f}, "
          f"Epsilon: {agent.epsilon:.4f}, Шаги: {step+1}")

    # Сохранение лучшей модели
    if avg_score > best_avg_score:
        best_avg_score = avg_score
        agent.save("dqn_2048_gym_best.pth")
        print(f"*** Новая лучшая модель сохранена с AvgScore: {best_avg_score:.2f} ***")

# --- Отображение результатов (код без изменений) ---
# ... (скопируйте ваш код для построения графиков из train_agent.py) ...