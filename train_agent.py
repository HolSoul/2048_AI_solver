import numpy as np
from game import Game # Наша игра
from ai_solver import DQNAgent, DEVICE # Наш агент на PyTorch и DEVICE
import matplotlib.pyplot as plt # Для графиков обучения

# --- Функции для состояния и награды ---
def get_state(board):
    flat_board = np.array(board).flatten()
    processed_board = np.zeros_like(flat_board, dtype=float)
    for i, tile_val in enumerate(flat_board):
        if tile_val == 0:
            processed_board[i] = 0.0
        else:
            processed_board[i] = np.log2(tile_val) / 11.0 # Нормализация на log2(2048)
    return processed_board.astype(np.float32)

def get_max_tile(board_raw):
    if not board_raw or not any(np.array(board_raw).flatten()): # Проверка, что доска не пуста и не состоит только из нулей
        return 0
    return np.max(board_raw)

def count_empty_cells(board_raw):
    return np.sum(np.array(board_raw) == 0)

def calculate_reward(score_before, score_after, 
                     board_before_raw, board_after_raw,
                     board_changed_by_move, game_over_flag):
    reward = 0

    # 1. Награда за увеличение счета (небольшой вес)
    score_increase = score_after - score_before
    reward += score_increase * 0.1 # Вес 0.1

    # 2. Награда, связанная с максимальной плиткой
    max_tile_before = get_max_tile(board_before_raw)
    max_tile_after = get_max_tile(board_after_raw)

    if max_tile_after > max_tile_before and max_tile_before > 0: # Условие max_tile_before > 0 добавлено, чтобы не было награды за появление первой плитки > 0
        # Значительный бонус за создание НОВОЙ максимальной плитки
        reward += (np.log2(max_tile_after + 1e-6) - np.log2(max_tile_before + 1e-6)) * 20 # Добавлен 1e-6 для стабильности log
    elif max_tile_after > 0 and max_tile_before == 0 : # Первая плитка на доске (не ноль)
        reward += np.log2(max_tile_after + 1e-6) * 5 # Начальный бонус за появление плитки

    # Небольшая награда за сам факт наличия высокой плитки (состояние)
    if max_tile_after > 0:
         reward += np.log2(max_tile_after + 1e-6) * 0.5 # Уменьшил вес, т.к. есть бонус за новую макс. плитку

    # 3. Бонус за пустые клетки
    empty_cells_after = count_empty_cells(board_after_raw)
    reward += empty_cells_after * 0.7 # Уменьшил вес

    # 4. Штрафы
    if not board_changed_by_move and not game_over_flag: 
        reward -= 15 # Увеличил штраф за бесполезный ход
    
    if game_over_flag: 
        if max_tile_after >= 2048:
            reward += 200 
        elif max_tile_after >= 1024:
            reward += 100 
        elif max_tile_after >= 512:
            reward += 50 
        elif max_tile_after >= 256:
            reward += 5 
        elif max_tile_after >= 128:
            reward -= 25 
        elif max_tile_after >= 64:
            reward -= 75 
        else:
            reward -= 150 # Большой штраф за проигрыш с низкими плитками
    
    return reward

# --- Параметры обучения ---
NUM_EPISODES = 2500 
MAX_STEPS_PER_EPISODE = 1000

# --- Инициализация ---
game_env_init = Game() 
state_size = game_env_init.size * game_env_init.size 
action_size = 4 

agent = DQNAgent(state_size, action_size, 
                 replay_buffer_size=20000, 
                 batch_size=128,          
                 learning_rate=0.00025,
                 gamma=0.99,
                 epsilon_decay=0.99995)

scores_history = []
avg_scores_history = []
max_tile_history = [] # Для отслеживания максимальной плитки

print(f"Начинаем обучение на {DEVICE}...")

# --- Цикл обучения ---
for episode in range(1, NUM_EPISODES + 1):
    game_env = Game() 
    # current_board_state_raw = game_env.board # Это копия, board меняется
    # current_score_val = game_env.score
    
    state = get_state(game_env.board) 
    total_reward_this_episode = 0
    episode_max_tile = 0

    for step in range(MAX_STEPS_PER_EPISODE):
        action = agent.act(state) 
        
        # Сохраняем состояние ДО хода
        board_before_raw = [list(row) for row in game_env.board] # Глубокая копия
        score_before_move = game_env.score

        animation_events = game_env.move(action) 
        
        board_changed = bool(animation_events) 
                                               
        next_board_state_raw = game_env.board # Это уже состояние ПОСЛЕ хода
        score_after_move = game_env.score
        game_is_over = game_env.is_game_over()
        
        current_max_on_board = get_max_tile(next_board_state_raw) # Используем функцию
        if current_max_on_board > episode_max_tile:
            episode_max_tile = current_max_on_board

        reward = calculate_reward(score_before_move, score_after_move, 
                                  board_before_raw, next_board_state_raw, 
                                  board_changed, game_is_over)
        total_reward_this_episode += reward
        
        next_state = get_state(next_board_state_raw)
        
        agent.remember(state, action, reward, next_state, game_is_over)
        
        state = next_state 
        
        if len(agent.memory) > agent.batch_size: 
            agent.replay()

        if game_is_over:
            break
            
    scores_history.append(game_env.score) 
    max_tile_history.append(episode_max_tile)
    avg_score = np.mean(scores_history[-100:]) 
    avg_max_tile = np.mean(max_tile_history[-100:])
    avg_scores_history.append(avg_score)

    print(f"Эпизод: {episode}/{NUM_EPISODES}, Счет: {game_env.score}, Макс.плитка: {episode_max_tile}, Сред.счет(100): {avg_score:.2f}, Сред.макс.пл(100): {avg_max_tile:.1f}, Epsilon: {agent.epsilon:.4f}, Шаги: {step+1}")

    if episode % 100 == 0: 
        agent.save(f"dqn_2048_pytorch_ep{episode}.pth")

# --- Отображение результатов --- 
# График 1: Счет
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(scores_history, label='Счет за эпизод', alpha=0.7)
plt.plot(avg_scores_history, label='Средний счет (100 эп.)', linewidth=2)
plt.xlabel("Эпизод")
plt.ylabel("Счет")
plt.title("История счета обучения DQN")
plt.legend()
plt.grid(True)

# График 2: Максимальная плитка
plt.subplot(1, 2, 2)
plt.plot(max_tile_history, label='Макс. плитка за эпизод', alpha=0.7, color='orange')
# Можно добавить скользящее среднее для макс. плитки, если нужно
# avg_max_tile_plot = [np.mean(max_tile_history[max(0,i-100):i+1]) for i in range(len(max_tile_history))]
# plt.plot(avg_max_tile_plot, label='Средняя макс. плитка (100 эп.)', linewidth=2, color='red')
plt.xlabel("Эпизод")
plt.ylabel("Максимальная плитка")
plt.title("История макс. плитки обучения DQN")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_history_pytorch.png")
print("График истории обучения сохранен в training_history_pytorch.png")
# plt.show() # Раскомментируйте, если хотите показать график сразу

print("Обучение завершено.") 