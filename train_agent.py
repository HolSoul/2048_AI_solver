import numpy as np
from game import Game # Наша игра
from ai_solver import DQNAgent, DEVICE # Наш агент на PyTorch и DEVICE
import matplotlib.pyplot as plt # Для графиков обучения

# --- Вспомогательные функции для стратегии "угол" ---
def get_max_tile_value_and_loc(board_raw):
    max_val = 0
    loc = (-1, -1) # По умолчанию, если доска пуста
    if not board_raw or not any(np.array(board_raw).flatten()):
        return 0, (-1,-1)
        
    for r_idx, row in enumerate(board_raw):
        for c_idx, val in enumerate(row):
            if val > max_val:
                max_val = val
                loc = (r_idx, c_idx)
    return max_val, loc

def calculate_line_monotonicity_and_smoothness(line_array):
    """Оценивает монотонность (невозрастание) и гладкость ряда.
       Более высокий балл лучше.
    """
    score = 0
    # 1. Бонус за монотонность (невозрастание)
    is_monotonic = True
    for i in range(len(line_array) - 1):
        # Штрафуем, если меньшая плитка стоит перед большей (кроме случая, когда меньшая - ноль)
        if line_array[i] < line_array[i+1] and line_array[i] != 0:
            is_monotonic = False
            score -= (np.log2(line_array[i+1] + 1e-6) - np.log2(line_array[i] + 1e-6)) * 2 # Штраф пропорционален "нарушению"
            # break # Можно раскомментировать для строгого штрафа за первое нарушение
    
    if is_monotonic:
        score += 5 # Базовый бонус за общую монотонность

    # 2. Бонус за "гладкость" и правильный порядок смежных плиток
    for i in range(len(line_array) - 1):
        val1 = line_array[i]
        val2 = line_array[i+1]
        if val1 > 0 and val2 > 0: # Обе плитки не нулевые
            if val1 >= val2: # Правильный порядок или равенство
                # Бонус пропорционален логарифму большей плитки
                score += np.log2(val1 + 1e-6) * 0.5 
                # Небольшой бонус, если val2 является степенью val1/2 (например, 128, 64)
                if val1 == val2 * 2:
                    score += 2
            else: # Неправильный порядок (меньшая перед большей)
                # Штраф уже учтен выше, но можно добавить дополнительный
                score -= np.log2(val2 + 1e-6) * 0.5 
        elif val1 > 0 and val2 == 0: # Плитка, за которой следует пустое место - хорошо
            score += 1
        # Если val1 == 0 и val2 > 0 - это уже покрывается штрафом за немонотонность
            
    # 3. Бонус за пустые клетки в конце ряда (если ряд заполняется слева направо)
    empty_at_end = 0
    for x in reversed(line_array):
        if x == 0:
            empty_at_end +=1
        else:
            break
    score += empty_at_end * 0.5
    
    return score

TARGET_CORNER_RC = (3, 0) # Левый нижний угол (строка 3, столбец 0)

# --- Функции для состояния и награды (старые и обновленная) ---
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
                     board_changed_by_move, game_over_flag,
                     action_taken 
                     ):
    reward = 0.0

    max_tile_before, _ = get_max_tile_value_and_loc(board_before_raw)
    max_tile_after, _ = get_max_tile_value_and_loc(board_after_raw)

    REWARD_MAX_TILE_INCREASE_LOG_MULTIPLIER = 10.0
    PENALTY_GAME_OVER = -100.0 
    REWARD_WIN_2048 = 200.0 # Награда за достижение 2048

    # 1. Награда за увеличение максимальной плитки на доске
    if max_tile_after > max_tile_before and max_tile_after > 0:
        # Награда пропорциональна логарифму новой максимальной плитки.
        # Это дает большие награды за достижение более высоких плиток.
        # Например, если новая макс. плитка 4 (log2(4)=2), награда +20.
        # Если новая макс. плитка 32 (log2(32)=5), награда +50.
        # Если новая макс. плитка 2048 (log2(2048)=11), награда +110.
        reward += np.log2(max_tile_after) * REWARD_MAX_TILE_INCREASE_LOG_MULTIPLIER
            
    # 2. Штраф/награда за окончание игры
    if game_over_flag:
        if max_tile_after >= 2048: # Условие победы
            reward += REWARD_WIN_2048
        else:
            # Штраф за проигрыш игры
            reward += PENALTY_GAME_OVER
            # Можно добавить более сложный штраф, зависящий от достигнутой плитки,
            # но пока ограничимся фиксированным значением для простоты.

    # Параметры (score_before, score_after, board_changed_by_move, action_taken)
    # пока не используются для упрощения, но остаются для возможного будущего расширения.
    return reward

# --- Параметры обучения ---
NUM_EPISODES = 3000 
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
                 epsilon_decay=0.99999)

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
                                  board_changed, game_is_over,
                                  action)
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