import numpy as np
from game import Game # Наша игра
from ai_solver import DQNAgent, DEVICE # Наш агент на PyTorch и DEVICE
import matplotlib.pyplot as plt # Для графиков обучения

# --- Вспомогательные функции для стратегии "угол" ---
def get_max_tile_value_and_loc(board_raw):
    max_val = 0 # Инициализируем max_val
    loc = (-1, -1) # По умолчанию, если доска пуста
    if not board_raw: # Проверка на пустой список board_raw
        return 0, (-1,-1)
        
    board_np_flat = np.array(board_raw).flatten()
    if not np.any(board_np_flat): # Проверка, что доска не состоит только из нулей
        return 0, (-1,-1)

    # Используем оригинальный итеративный способ для поиска максимальной плитки и ее локации,
    # чтобы обеспечить консистентность с логикой выбора угла, если есть несколько одинаковых макс. плиток.
    for r_idx, row in enumerate(board_raw):
        for c_idx, val in enumerate(row):
            if val > max_val:
                max_val = val
                loc = (r_idx, c_idx)
    
    if max_val == 0: # Если после итерации максимальное значение все еще 0
        return 0, (-1,-1)
    return max_val, loc


def calculate_line_monotonicity_and_smoothness(line_array_raw):
    """Оценивает монотонность (невозрастание) и гладкость ряда.
       Более высокий балл лучше.
    """
    line_array = np.array(line_array_raw) # Убедимся, что это numpy array

    score = 0.0 # Используем float для счета
    # 1. Бонус за монотонность (невозрастание слева направо)
    is_monotonic_decreasing = True
    for i in range(len(line_array) - 1):
        # Штрафуем, если меньшая плитка стоит перед большей (обе не нулевые)
        if line_array[i] != 0 and line_array[i+1] != 0 and line_array[i] < line_array[i+1]:
            is_monotonic_decreasing = False
            # Штраф пропорционален логарифмической разнице "нарушения"
            score -= (np.log2(line_array[i+1] + 1e-9) - np.log2(line_array[i] + 1e-9)) * 1.5 
            # break # Можно раскомментировать для строгого штрафа за первое нарушение
    
    if is_monotonic_decreasing:
        score += 2.0 # Базовый бонус за общую монотонность

    # 2. Бонус за "гладкость" и правильный порядок смежных плиток
    for i in range(len(line_array) - 1):
        val1 = line_array[i]
        val2 = line_array[i+1]
        if val1 > 0 and val2 > 0: # Обе плитки не нулевые
            if val1 >= val2: # Правильный порядок или равенство
                score += np.log2(val1 + 1e-9) * 0.25 
                if val1 == val2 * 2: # Например, 128, 64
                    score += 1.0
            # else: # Неправильный порядок (меньшая перед большей) - штраф уже учтен выше
                # pass
        elif val1 > 0 and val2 == 0: # Плитка, за которой следует пустое место - хорошо для края
            score += 0.5
            
    # 3. Бонус за пустые клетки в конце ряда (если ряд заполняется слева направо)
    empty_at_end = 0
    for x_val in reversed(line_array): # Идем справа налево
        if x_val == 0:
            empty_at_end +=1
        else:
            break
    score += empty_at_end * 0.25
    
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
                     action_taken # action_taken пока не используется
                     ):
    reward = 0.0

    max_tile_before, loc_max_tile_before = get_max_tile_value_and_loc(board_before_raw)
    max_tile_after, loc_max_tile_after = get_max_tile_value_and_loc(board_after_raw)
    num_empty_after = count_empty_cells(board_after_raw)
    board_np_after = np.array(board_after_raw)
    board_np_before = np.array(board_before_raw) # Понадобится для штрафа за уход из угла

    # --- Константы для наград и штрафов ---
    REWARD_SCORE_INCREASE_MULTIPLIER = 0.01  # Небольшой бонус за очки
    REWARD_MAX_TILE_INCREASE_LOG_MULTIPLIER = 20.0 # Значительно важнее достижение новых плиток
    PENALTY_GAME_OVER = -300.0 # Существенный штраф за проигрыш (модифицируем масштабирование)
    REWARD_WIN_2048 = 1500.0   # Большая награда за победу (увеличим)
    PENALTY_INVALID_MOVE = -50.0 # Штраф за ход, не изменивший доску (если это не конец игры)
    REWARD_EMPTY_CELLS_MULTIPLIER = 0.5  # За каждую пустую клетку (уменьшим)
    
    # Связанные с углом
    REWARD_MAX_TILE_IN_CORNER = 80.0  # Бонус, если максимальная плитка в целевом углу (увеличим)
    WEIGHT_MAX_TILE_CORNER_LOG = True # Умножать бонус на log2(max_tile_after)?
    PENALTY_MAX_TILE_LEFT_CORNER = -70.0 # Штраф, если макс. плитка ушла из угла или угол ухудшился
    REWARD_LOCKED_CORNER_BONUS = 20.0    # Бонус, если макс. плитка в углу "заперта" большими соседями
    WEIGHT_LOCKED_CORNER_LOG = True    # Умножать бонус "запертости" на log2(max_tile_after)?

    # Эвристики доски
    REWARD_MONOTONICITY_OVERALL_WEIGHT = 0.3 # Общий вес для монотонности/гладкости (увеличим)
    REWARD_SMOOTHNESS_WEIGHT = 0.1 # Дополнительный вес для гладкости (соседние одинаковые или x, x/2)
    REWARD_POTENTIAL_MERGES_MULTIPLIER = 0.5 # За каждую пару потенциальных слияний
    
    epsilon_log = 1e-9 # для стабильности логарифмов

    # 0. Штраф за невалидный ход (если доска не изменилась и игра не окончена)
    if not board_changed_by_move and not game_over_flag:
        reward += PENALTY_INVALID_MOVE
        return np.clip(reward, -400.0, 1700.0) # Обновим границы клиппинга

    # 1. Награда за увеличение счета
    if score_after > score_before:
        reward += (score_after - score_before) * REWARD_SCORE_INCREASE_MULTIPLIER

    # 2. Награда за увеличение максимальной плитки
    if max_tile_after > max_tile_before and max_tile_after > 0:
        increase_bonus = np.log2(max_tile_after + epsilon_log) * REWARD_MAX_TILE_INCREASE_LOG_MULTIPLIER
        if max_tile_after > 2 and (max_tile_after & (max_tile_after - 1) == 0): # Степень двойки
             increase_bonus += np.log2(max_tile_after + epsilon_log) * (REWARD_MAX_TILE_INCREASE_LOG_MULTIPLIER / 1.5) # Усилим бонус за степень двойки
        reward += increase_bonus
            
    # 3. Награда за количество пустых клеток
    reward += num_empty_after * REWARD_EMPTY_CELLS_MULTIPLIER

    # 4. Логика УГЛА (TARGET_CORNER_RC)
    current_target_corner = TARGET_CORNER_RC 
    # 4a. Бонус за максимальную плитку в целевом углу
    if loc_max_tile_after == current_target_corner and max_tile_after > 0:
        corner_bonus = REWARD_MAX_TILE_IN_CORNER
        if WEIGHT_MAX_TILE_CORNER_LOG:
            corner_bonus *= np.log2(max_tile_after + epsilon_log) 
        reward += corner_bonus

        # 4b. Бонус за "запертую" максимальную плитку в углу
        # (только если макс плитка уже в углу)
        is_locked = True
        # Предполагаем TARGET_CORNER_RC = (3,0) - левый нижний
        # Сосед справа board_np_after[3,1], сосед сверху board_np_after[2,0]
        # Проверяем правого соседа (если он существует)
        if current_target_corner[1] + 1 < board_np_after.shape[1]:
            neighbor_right = board_np_after[current_target_corner[0], current_target_corner[1] + 1]
            if not (neighbor_right == 0 or neighbor_right >= max_tile_after): # Должен быть 0 или больше/равен
                is_locked = False
        # Проверяем верхнего соседа (если он существует)
        if current_target_corner[0] - 1 >= 0:
            neighbor_up = board_np_after[current_target_corner[0] - 1, current_target_corner[1]]
            if not (neighbor_up == 0 or neighbor_up >= max_tile_after): # Должен быть 0 или больше/равен
                is_locked = False
        
        if is_locked:
            locked_bonus = REWARD_LOCKED_CORNER_BONUS
            if WEIGHT_LOCKED_CORNER_LOG:
                locked_bonus *= np.log2(max_tile_after + epsilon_log)
            reward += locked_bonus

    # 4c. Штраф, если максимальная плитка покинула угол или угол ухудшился
    if loc_max_tile_before == current_target_corner and max_tile_before > 0: # Если раньше макс. была в углу
        tile_in_corner_after = board_np_after[current_target_corner[0], current_target_corner[1]]
        # Условие 1: Максимальная плитка вообще ушла из угла
        # Условие 2: Максимальная плитка осталась та же, но ее позиция изменилась (т.е. она ушла из угла)
        # Условие 3: В углу теперь плитка МЕНЬШЕ, чем была максимальная плитка до этого (которая была в углу)
        if loc_max_tile_after != current_target_corner or \
           (loc_max_tile_after == current_target_corner and max_tile_after < max_tile_before) or \
           tile_in_corner_after < max_tile_before:
            penalty_val = PENALTY_MAX_TILE_LEFT_CORNER
            # Можно сделать штраф зависимым от величины ушедшей плитки
            # penalty_val *= np.log2(max_tile_before + epsilon_log)
            reward += penalty_val
    
    # 5. Эвристики монотонности и гладкости
    grid_score = 0.0
    # Монотонность вдоль строк (ожидаем убывание к правому краю для (3,0) или (0,0))
    for r in range(board_np_after.shape[0]):
        row = board_np_after[r, :]
        # Дадим больший вес строке, где находится целевой угол
        weight = 1.0 if r == current_target_corner[0] else 0.7 
        grid_score += calculate_line_monotonicity_and_smoothness(row) * weight

    # Монотонность вдоль столбцов
    for c_col_idx in range(board_np_after.shape[1]): 
        col = board_np_after[:, c_col_idx]
        weight = 1.0 if c_col_idx == current_target_corner[1] else 0.7
        if current_target_corner[0] == 0: # Если угол верхний 
            grid_score += calculate_line_monotonicity_and_smoothness(col) * weight
        elif current_target_corner[0] == board_np_after.shape[0] - 1: # Если угол нижний
            grid_score += calculate_line_monotonicity_and_smoothness(col[::-1]) * weight
    reward += grid_score * REWARD_MONOTONICITY_OVERALL_WEIGHT

    # Дополнительная оценка гладкости (штраф за большие разрывы между соседними плитками)
    smoothness_penalty = 0.0
    for r_idx in range(board_np_after.shape[0]):
        for c_idx_smooth in range(board_np_after.shape[1]): # переименовал 'c_idx' в 'c_idx_smooth'
            current_val = board_np_after[r_idx,c_idx_smooth]
            if current_val == 0: continue
            
            # Проверяем соседа справа
            if c_idx_smooth + 1 < board_np_after.shape[1]:
                right_val = board_np_after[r_idx, c_idx_smooth+1]
                if right_val != 0 and abs(np.log2(current_val+epsilon_log) - np.log2(right_val+epsilon_log)) > 1.0: 
                    smoothness_penalty -= abs(np.log2(current_val+epsilon_log) - np.log2(right_val+epsilon_log)) 
            # Проверяем соседа снизу
            if r_idx + 1 < board_np_after.shape[0]:
                down_val = board_np_after[r_idx+1, c_idx_smooth]
                if down_val != 0 and abs(np.log2(current_val+epsilon_log) - np.log2(down_val+epsilon_log)) > 1.0:
                    smoothness_penalty -= abs(np.log2(current_val+epsilon_log) - np.log2(down_val+epsilon_log))
    reward += smoothness_penalty * REWARD_SMOOTHNESS_WEIGHT

    # 5b. Бонус за потенциальные слияния
    potential_merges = 0
    for r_idx in range(board_np_after.shape[0]):
        for c_idx_pm in range(board_np_after.shape[1]): #pm для potential merge
            current_val_pm = board_np_after[r_idx, c_idx_pm]
            if current_val_pm == 0: continue
            # Проверяем соседа справа
            if c_idx_pm + 1 < board_np_after.shape[1]:
                if board_np_after[r_idx, c_idx_pm + 1] == current_val_pm:
                    potential_merges += 1
            # Проверяем соседа снизу
            if r_idx + 1 < board_np_after.shape[0]:
                if board_np_after[r_idx + 1, c_idx_pm] == current_val_pm:
                    potential_merges += 1
    reward += potential_merges * REWARD_POTENTIAL_MERGES_MULTIPLIER

    # 6. Штраф/награда за окончание игры
    if game_over_flag:
        if max_tile_after >= 2048:
            reward += REWARD_WIN_2048
        else:
            # Штраф более чувствителен к низкой максимальной плитке при проигрыше
            penalty_game_over_scaled = PENALTY_GAME_OVER
            if max_tile_after > 0: 
                 # Чем выше плитка, тем меньше штраф (ближе к 0)
                 # Чем плитка ближе к 2, тем штраф ближе к PENALTY_GAME_OVER
                 penalty_scale_factor = (np.log2(2048.0 + epsilon_log) - np.log2(max_tile_after + epsilon_log)) / (np.log2(2048.0 + epsilon_log) - np.log2(2.0 + epsilon_log) + epsilon_log)
                 penalty_game_over_scaled *= penalty_scale_factor
            else: # Проигрыш с пустой доской или только с плиткой 2 - максимальный штраф
                 penalty_game_over_scaled = PENALTY_GAME_OVER * 1.2 # Еще чуть больше

            reward += penalty_game_over_scaled
            if max_tile_after < 32 : # Дополнительный сильный штраф, если проиграли с очень низкой плиткой (меньше 32)
                reward -= 100.0 
            elif max_tile_after < 128: # Мягче штраф, если меньше 128
                reward -= 50.0

    return np.clip(reward, -400.0, 1700.0) # Обновим границы клиппинга

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

# --- Переменная для отслеживания лучшего результата ---
best_avg_score = -float('inf') # Инициализируем очень маленьким числом

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

    # Сохранение лучшей модели по среднему счету
    if avg_score > best_avg_score:
        best_avg_score = avg_score
        agent.save(f"dqn_2048_pytorch_best_avg_score.pth")
        print(f"*** Новая лучшая модель сохранена с AvgScore: {best_avg_score:.2f} в эпизоде {episode} ***")

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