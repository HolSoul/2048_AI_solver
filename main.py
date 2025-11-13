import pygame
import sys
import os # Для проверки существования файла модели
import numpy as np # Для get_state
from game import Game
# from ai_solver import AISolver # Старый placeholder, если был
from ai_solver import DQNAgent, DEVICE, QNetwork # Наш агент и устройство, импортируем также QNetwork и torch из ai_solver, если они там
import datetime # Для именования лог файлов

# Попытка импортировать torch глобально в main.py для проверки в initialize_ai_agent
if 'torch' not in globals():
    try:
        import torch
        print("(main.py) PyTorch импортирован для проверки.")
    except ImportError:
        print("(main.py) PyTorch не найден. ИИ не будет доступен.")
        torch = None 

# --- Функции, скопированные/адаптированные из train_agent.py для отладки вознаграждений ---
TARGET_CORNER_RC = (3, 0) # Убедитесь, что это тот же угол, что и при обучении

def get_max_tile_value_and_loc(board_raw): # Копия из train_agent.py
    max_val = 0 
    loc = (-1, -1) 
    if not board_raw: 
        return 0, (-1,-1)
    board_np_flat = np.array(board_raw).flatten()
    if not np.any(board_np_flat): 
        return 0, (-1,-1)
    for r_idx, row in enumerate(board_raw):
        for c_idx, val_in_row in enumerate(row): # val переименовано в val_in_row
            if val_in_row > max_val:
                max_val = val_in_row
                loc = (r_idx, c_idx)
    if max_val == 0: 
        return 0, (-1,-1)
    return max_val, loc

def count_empty_cells(board_raw): # Копия из train_agent.py
    return np.sum(np.array(board_raw) == 0)

def calculate_line_monotonicity_and_smoothness(line_array_raw): # Копия из train_agent.py
    line_array = np.array(line_array_raw) 
    score = 0.0 
    is_monotonic_decreasing = True
    for i in range(len(line_array) - 1):
        if line_array[i] != 0 and line_array[i+1] != 0 and line_array[i] < line_array[i+1]:
            is_monotonic_decreasing = False
            score -= (np.log2(line_array[i+1] + 1e-9) - np.log2(line_array[i] + 1e-9)) * 1.5 
    if is_monotonic_decreasing:
        score += 2.0 
    for i in range(len(line_array) - 1):
        val1 = line_array[i]
        val2 = line_array[i+1]
        if val1 > 0 and val2 > 0: 
            if val1 >= val2: 
                score += np.log2(val1 + 1e-9) * 0.25 
                if val1 == val2 * 2: 
                    score += 1.0
        elif val1 > 0 and val2 == 0: 
            score += 0.5
    empty_at_end = 0
    for x_val in reversed(line_array): 
        if x_val == 0:
            empty_at_end +=1
        else:
            break
    score += empty_at_end * 0.25
    return score

# Заменяем calculate_reward_for_debug на полную копию из train_agent.py
def calculate_reward_for_debug(score_before, score_after, 
                               board_before_raw, board_after_raw,
                               board_changed_by_move, game_over_flag):
    reward = 0.0

    max_tile_before, loc_max_tile_before = get_max_tile_value_and_loc(board_before_raw)
    max_tile_after, loc_max_tile_after = get_max_tile_value_and_loc(board_after_raw)
    num_empty_after = count_empty_cells(board_after_raw)
    board_np_after = np.array(board_after_raw)
    board_np_before = np.array(board_before_raw) 

    # --- Константы для наград и штрафов (должны быть идентичны train_agent.py) ---
    REWARD_SCORE_INCREASE_MULTIPLIER = 0.01  
    REWARD_MAX_TILE_INCREASE_LOG_MULTIPLIER = 20.0 
    PENALTY_GAME_OVER = -300.0 
    REWARD_WIN_2048 = 1500.0   
    PENALTY_INVALID_MOVE = -50.0 
    REWARD_EMPTY_CELLS_MULTIPLIER = 0.5  
    
    REWARD_MAX_TILE_IN_CORNER = 80.0  
    WEIGHT_MAX_TILE_CORNER_LOG = True 
    PENALTY_MAX_TILE_LEFT_CORNER = -70.0 
    REWARD_LOCKED_CORNER_BONUS = 20.0    
    WEIGHT_LOCKED_CORNER_LOG = True    

    REWARD_MONOTONICITY_OVERALL_WEIGHT = 0.3 
    REWARD_SMOOTHNESS_WEIGHT = 0.1 
    REWARD_POTENTIAL_MERGES_MULTIPLIER = 0.5 
    
    epsilon_log = 1e-9 

    if not board_changed_by_move and not game_over_flag:
        reward += PENALTY_INVALID_MOVE
        return np.clip(reward, -400.0, 1700.0) 

    if score_after > score_before:
        reward += (score_after - score_before) * REWARD_SCORE_INCREASE_MULTIPLIER

    if max_tile_after > max_tile_before and max_tile_after > 0:
        increase_bonus = np.log2(max_tile_after + epsilon_log) * REWARD_MAX_TILE_INCREASE_LOG_MULTIPLIER
        if max_tile_after > 2 and (max_tile_after & (max_tile_after - 1) == 0): 
             increase_bonus += np.log2(max_tile_after + epsilon_log) * (REWARD_MAX_TILE_INCREASE_LOG_MULTIPLIER / 1.5) 
        reward += increase_bonus
            
    reward += num_empty_after * REWARD_EMPTY_CELLS_MULTIPLIER

    current_target_corner = TARGET_CORNER_RC 
    if loc_max_tile_after == current_target_corner and max_tile_after > 0:
        corner_bonus = REWARD_MAX_TILE_IN_CORNER
        if WEIGHT_MAX_TILE_CORNER_LOG:
            corner_bonus *= np.log2(max_tile_after + epsilon_log) 
        reward += corner_bonus

        is_locked = True
        if current_target_corner[1] + 1 < board_np_after.shape[1]:
            neighbor_right = board_np_after[current_target_corner[0], current_target_corner[1] + 1]
            if not (neighbor_right == 0 or neighbor_right >= max_tile_after): 
                is_locked = False
        if current_target_corner[0] - 1 >= 0:
            neighbor_up = board_np_after[current_target_corner[0] - 1, current_target_corner[1]]
            if not (neighbor_up == 0 or neighbor_up >= max_tile_after): 
                is_locked = False
        
        if is_locked:
            locked_bonus = REWARD_LOCKED_CORNER_BONUS
            if WEIGHT_LOCKED_CORNER_LOG:
                locked_bonus *= np.log2(max_tile_after + epsilon_log)
            reward += locked_bonus

    if loc_max_tile_before == current_target_corner and max_tile_before > 0: 
        tile_in_corner_after = board_np_after[current_target_corner[0], current_target_corner[1]]
        if loc_max_tile_after != current_target_corner or \
           (loc_max_tile_after == current_target_corner and max_tile_after < max_tile_before) or \
           tile_in_corner_after < max_tile_before:
            penalty_val = PENALTY_MAX_TILE_LEFT_CORNER
            reward += penalty_val
    
    grid_score = 0.0
    for r in range(board_np_after.shape[0]):
        row_data = board_np_after[r, :] # Renamed variable to avoid conflict
        weight = 1.0 if r == current_target_corner[0] else 0.7 
        grid_score += calculate_line_monotonicity_and_smoothness(row_data) * weight

    for c_col_idx in range(board_np_after.shape[1]): 
        col_data = board_np_after[:, c_col_idx] # Renamed variable to avoid conflict
        weight = 1.0 if c_col_idx == current_target_corner[1] else 0.7
        if current_target_corner[0] == 0: 
            grid_score += calculate_line_monotonicity_and_smoothness(col_data) * weight
        elif current_target_corner[0] == board_np_after.shape[0] - 1: 
            grid_score += calculate_line_monotonicity_and_smoothness(col_data[::-1]) * weight
    reward += grid_score * REWARD_MONOTONICITY_OVERALL_WEIGHT

    smoothness_penalty = 0.0
    for r_idx_smooth, row_smooth in enumerate(board_np_after): # Iterate directly over rows for clarity
        for c_idx_smooth, current_val_smooth in enumerate(row_smooth): # current_val renamed
            if current_val_smooth == 0: continue
            if c_idx_smooth + 1 < board_np_after.shape[1]:
                right_val = board_np_after[r_idx_smooth, c_idx_smooth + 1]
                if right_val != 0 and abs(np.log2(current_val_smooth + epsilon_log) - np.log2(right_val + epsilon_log)) > 1.0: 
                    smoothness_penalty -= abs(np.log2(current_val_smooth + epsilon_log) - np.log2(right_val + epsilon_log)) 
            if r_idx_smooth + 1 < board_np_after.shape[0]:
                down_val = board_np_after[r_idx_smooth + 1, c_idx_smooth]
                if down_val != 0 and abs(np.log2(current_val_smooth + epsilon_log) - np.log2(down_val + epsilon_log)) > 1.0:
                    smoothness_penalty -= abs(np.log2(current_val_smooth + epsilon_log) - np.log2(down_val + epsilon_log))
    reward += smoothness_penalty * REWARD_SMOOTHNESS_WEIGHT

    potential_merges = 0
    for r_idx_pm in range(board_np_after.shape[0]):
        for c_idx_pm in range(board_np_after.shape[1]): 
            current_val_pm_merge = board_np_after[r_idx_pm, c_idx_pm] # Renamed variable
            if current_val_pm_merge == 0: continue
            if c_idx_pm + 1 < board_np_after.shape[1]:
                if board_np_after[r_idx_pm, c_idx_pm + 1] == current_val_pm_merge:
                    potential_merges += 1
            if r_idx_pm + 1 < board_np_after.shape[0]:
                if board_np_after[r_idx_pm + 1, c_idx_pm] == current_val_pm_merge:
                    potential_merges += 1
    reward += potential_merges * REWARD_POTENTIAL_MERGES_MULTIPLIER

    if game_over_flag:
        if max_tile_after >= 2048:
            reward += REWARD_WIN_2048
        else:
            penalty_game_over_scaled = PENALTY_GAME_OVER
            if max_tile_after > 0: 
                 penalty_scale_factor = (np.log2(2048.0 + epsilon_log) - np.log2(max_tile_after + epsilon_log)) / (np.log2(2048.0 + epsilon_log) - np.log2(2.0 + epsilon_log) + epsilon_log)
                 penalty_game_over_scaled *= penalty_scale_factor
            else: 
                 penalty_game_over_scaled = PENALTY_GAME_OVER * 1.2 

            reward += penalty_game_over_scaled
            if max_tile_after < 32 : 
                reward -= 100.0 
            elif max_tile_after < 128: 
                reward -= 50.0

    return np.clip(reward, -400.0, 1700.0)

# Копируем get_state из train_agent.py (или можно импортировать, если вынести в утилиты)
def get_state(board):
    flat_board = np.array(board).flatten()
    processed_board = np.zeros_like(flat_board, dtype=float)
    for i, tile_val in enumerate(flat_board):
        if tile_val == 0:
            processed_board[i] = 0.0
        else:
            # Нормализация на log2(2048) = 11.0. Убедитесь, что это соответствует обучению.
            processed_board[i] = np.log2(tile_val) / 11.0 
    return processed_board.astype(np.float32)

# Инициализация Pygame
pygame.init()
pygame.font.init() # Инициализация модуля для работы со шрифтами

# Константы
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 500 # Немного больше для отображения счета
GRID_SIZE = 4
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE
GRID_LINE_WIDTH = 6
BACKGROUND_COLOR = (187, 173, 160)
GRID_COLOR = (205, 193, 180) # Не используется напрямую для сетки, но как фон ячеек
FONT_COLOR = (119, 110, 101)
SCORE_FONT_COLOR = (238, 228, 218)
GAME_OVER_FONT_COLOR = (119, 110, 101)
AI_TEXT_COLOR = (255, 0, 0) # Цвет для текста "AI Active"


# Цвета для плиток (значение: (цвет_плитки, цвет_текста))
TILE_COLORS = {
    0: ((205, 193, 180), FONT_COLOR),  # Пустая ячейка
    2: ((238, 228, 218), FONT_COLOR),
    4: ((237, 224, 200), FONT_COLOR),
    8: ((242, 177, 121), SCORE_FONT_COLOR),
    16: ((245, 149, 99), SCORE_FONT_COLOR),
    32: ((246, 124, 95), SCORE_FONT_COLOR),
    64: ((246, 94, 59), SCORE_FONT_COLOR),
    128: ((237, 207, 114), SCORE_FONT_COLOR),
    256: ((237, 204, 97), SCORE_FONT_COLOR),
    512: ((237, 200, 80), SCORE_FONT_COLOR),
    1024: ((237, 197, 63), SCORE_FONT_COLOR),
    2048: ((237, 194, 46), SCORE_FONT_COLOR),
    # Добавьте больше цветов для больших значений, если нужно
}

# Шрифты
try:
    TILE_FONT = pygame.font.SysFont("arial", 40, bold=True)
    SCORE_LABEL_FONT = pygame.font.SysFont("arial", 20, bold=True)
    SCORE_FONT = pygame.font.SysFont("arial", 25, bold=True)
    GAME_OVER_FONT = pygame.font.SysFont("arial", 40, bold=True)
    AI_STATUS_FONT = pygame.font.SysFont("arial", 18, bold=True)
except pygame.error as e:
    print(f"Ошибка загрузки шрифта Arial: {e}. Используется шрифт по умолчанию.")
    TILE_FONT = pygame.font.Font(None, 55) # Шрифт по умолчанию, если Arial недоступен
    SCORE_LABEL_FONT = pygame.font.Font(None, 30)
    SCORE_FONT = pygame.font.Font(None, 35)
    GAME_OVER_FONT = pygame.font.Font(None, 60)
    AI_STATUS_FONT = pygame.font.Font(None, 25)


# Настройка экрана
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2048 Game")

# Анимации
ANIMATION_DURATION_MS = 150 # длительность анимации в миллисекундах
active_animations = [] # Теперь будет содержать более сложные объекты анимации

# --- Переменные для управления ИИ ---
ai_agent = None
ai_active = False
ai_model_loaded = False
DEFAULT_MODEL_FILENAME = "dqn_2048_pytorch_best_avg_score.pth" # Или конкретный файл, например dqn_2048_pytorch_ep2000.pth
AI_MOVE_DELAY_MS = 100 # Задержка между ходами ИИ в миллисекундах

def initialize_ai_agent(filename=DEFAULT_MODEL_FILENAME):
    global ai_agent, ai_model_loaded, torch # Указываем, что torch - глобальная переменная
    if torch is None: 
        print("PyTorch не доступен. ИИ не может быть загружен.")
        ai_model_loaded = False
        return
    try:
        game_temp = Game() # Для получения state_size
        state_size = game_temp.size * game_temp.size
        action_size = 4 
        
        loaded_agent = DQNAgent(state_size, action_size) # Создаем экземпляр
        # Путь к модели формируется внутри agent.load() на основе его model_save_path
        loaded_agent.load(filename) # Загружаем веса
        loaded_agent.policy_net.eval() # Важно перевести в режим оценки
        if hasattr(loaded_agent, 'target_net'): # Если есть target_net, его тоже в eval
             loaded_agent.target_net.eval()
        
        ai_agent = loaded_agent
        ai_model_loaded = True
        print(f"Агент ИИ успешно загружен с {filename} на устройстве {DEVICE}.")
    except Exception as e:
        print(f"Ошибка при загрузке/инициализации агента ИИ: {e}")
        ai_model_loaded = False

# --- Вспомогательные функции для анимаций ---
def rc_to_pixels(r, c):
    return c * CELL_SIZE, r * CELL_SIZE

def add_animation_from_event(event):
    """Добавляет анимацию в active_animations на основе события из game.py"""
    now = pygame.time.get_ticks()
    anim_obj = {
        'type': event['type'],
        'start_time': now,
        'progress': 0
    }

    if event['type'] == 'appear':
        anim_obj.update({
            'pos_rc': event['pos_rc'],
            'value': event['value']
        })
    elif event['type'] == 'move':
        anim_obj.update({
            'from_rc': event['from_rc'],
            'to_rc': event['to_rc'],
            'value': event['value']
        })
    elif event['type'] == 'merge':
        # Слияние порождает несколько визуальных частей:
        # 1. Плитка, которая движется (from_rc2 -> to_rc)
        # 2. Плитка, которая "принимает" слияние и меняет значение (to_rc)
        
        active_animations.append({
            'type': 'merge_moving_tile',
            'start_time': now,
            'progress': 0,
            'from_rc': event['from_rc2'],
            'to_rc': event['to_rc'], 
            'value': event['original_value'] # Движется плитка со старым значением
        })
        # Анимация для целевой плитки (которая увеличится/изменит значение)
        active_animations.append({
            'type': 'merge_target_tile',
            'start_time': now, # Начнется одновременно, но эффект может быть в конце
            'progress': 0,
            'pos_rc': event['to_rc'],
            'original_value': event['original_value'], # Значение до слияния (то, что было на to_rc)
            'merged_value': event['merged_value'],
            'from_rc1_original_pos': event['from_rc1'] # Чтобы знать, что там было до слияния
        })
        return # Merge обрабатывается специально, не добавляем общий anim_obj

    active_animations.append(anim_obj)


def get_tile_colors(value):
    # Для значений больше 2048, можно использовать цвет для 2048 или определить новые
    return TILE_COLORS.get(value, TILE_COLORS[2048]) 

def draw_board(board_data, current_score):
    screen.fill(BACKGROUND_COLOR)

    # Отображение счета
    score_label_surface = SCORE_LABEL_FONT.render("SCORE", True, FONT_COLOR)
    score_value_surface = SCORE_FONT.render(str(current_score), True, SCORE_FONT_COLOR)
    
    screen.blit(score_label_surface, (20, SCREEN_WIDTH + 10))
    screen.blit(score_value_surface, (20, SCREEN_WIDTH + 35))

    # Отображение статуса ИИ
    if ai_active and ai_model_loaded:
        status_text = "AI ACTIVE (M to toggle)"
        status_color = AI_TEXT_COLOR
    elif ai_active and not ai_model_loaded:
        status_text = "AI LOADING FAILED (M to toggle)"
        status_color = AI_TEXT_COLOR
    else:
        status_text = "Human Player (M for AI)"
        status_color = FONT_COLOR
    ai_status_surface = AI_STATUS_FONT.render(status_text, True, status_color)
    screen.blit(ai_status_surface, (SCREEN_WIDTH - ai_status_surface.get_width() - 10, SCREEN_WIDTH + 15))

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            tile_value = board_data[r][c]
            tile_color, text_color = get_tile_colors(tile_value)
            
            rect_x = c * CELL_SIZE
            rect_y = r * CELL_SIZE
            
            # ЕСЛИ ЯЧЕЙКА УЧАСТВУЕТ В АКТИВНОЙ АНИМАЦИИ (движение из нее, в нее, или слияние на ней)
            # то ее не рисуем здесь стандартно, она будет нарисована в цикле анимаций.
            # Вместо этого рисуем под ней пустую ячейку (или фон).
            is_involved_in_animation = False
            for anim in active_animations:
                if anim['type'] == 'move' and (anim['from_rc'] == (r,c) or anim['to_rc'] == (r,c)):
                    is_involved_in_animation = True; break
                if anim['type'] == 'appear' and anim['pos_rc'] == (r,c):
                    is_involved_in_animation = True; break
                if anim['type'] == 'merge_moving_tile' and (anim['from_rc'] == (r,c) or anim['to_rc'] == (r,c)):
                    is_involved_in_animation = True; break
                if anim['type'] == 'merge_target_tile' and anim['pos_rc'] == (r,c):
                    # Также, если эта ячейка была одной из исходных для слияния (from_rc1)
                    if anim['from_rc1_original_pos'] == (r,c):
                        is_involved_in_animation = True; break

            if is_involved_in_animation:
                # Рисуем фон ячейки, так как анимированная плитка будет поверх
                pygame.draw.rect(screen, TILE_COLORS[0][0], (rect_x, rect_y, CELL_SIZE, CELL_SIZE))
                continue

            # Рисуем ячейку (фон плитки), если она не анимируется
            pygame.draw.rect(screen, tile_color, (rect_x, rect_y, CELL_SIZE, CELL_SIZE))
            
            # Рисуем границы/сетку тоньше вокруг каждой плитки, чтобы было похоже на отступы
            # Внешняя рамка поля будет за счет фона окна, если он немного больше поля
            # Для внутренних линий между ячейками:
            if c > 0: # Вертикальная линия слева
                 pygame.draw.line(screen, BACKGROUND_COLOR, (rect_x, rect_y), (rect_x, rect_y + CELL_SIZE), GRID_LINE_WIDTH)
            if r > 0: # Горизонтальная линия сверху
                 pygame.draw.line(screen, BACKGROUND_COLOR, (rect_x, rect_y), (rect_x + CELL_SIZE, rect_y), GRID_LINE_WIDTH)


            if tile_value != 0:
                text_surface = TILE_FONT.render(str(tile_value), True, text_color)
                text_rect = text_surface.get_rect(center=(rect_x + CELL_SIZE / 2, 
                                                          rect_y + CELL_SIZE / 2))
                screen.blit(text_surface, text_rect)

    # Рисуем активные анимации
    # Важно: active_animations может изменяться во время итерации, если анимация завершается
    # Поэтому создаем копию для итерации
    for anim in list(active_animations): # Итерируемся по копии
        progress = anim['progress']

        if anim['type'] == 'appear':
            r, c = anim['pos_rc']
            value = anim['value'] # Исправлено: ключ должен быть 'value'
            progress = anim['progress']

            base_tile_color, text_color = get_tile_colors(value)
            
            # Анимация: увеличение размера
            current_size = CELL_SIZE * progress
            offset = (CELL_SIZE - current_size) / 2
            
            anim_rect_x = c * CELL_SIZE + offset
            anim_rect_y = r * CELL_SIZE + offset

            pygame.draw.rect(screen, base_tile_color, (anim_rect_x, anim_rect_y, current_size, current_size))

            if progress > 0.5: # Начинаем показывать текст, когда плитка достаточно большая
                # Масштабируем шрифт или показываем, когда размер близок к полному
                # Для простоты, покажем текст, когда progress > 0.5 и используем стандартный размер шрифта
                # Чтобы текст правильно центрировался, его нужно рисовать относительно полного размера ячейки
                text_surface = TILE_FONT.render(str(value), True, text_color)
                text_rect = text_surface.get_rect(center=(c * CELL_SIZE + CELL_SIZE / 2, 
                                                          r * CELL_SIZE + CELL_SIZE / 2))
                # Можно добавить альфа-канал для плавного появления текста, но пока упростим
                if current_size > CELL_SIZE * 0.7 : # Только если плитка достаточно видна
                    screen.blit(text_surface, text_rect)

        elif anim['type'] == 'move':
            r_from, c_from = anim['from_rc']
            r_to, c_to = anim['to_rc']
            val = anim['value']

            start_x, start_y = rc_to_pixels(r_from, c_from)
            end_x, end_y = rc_to_pixels(r_to, c_to)

            current_x = start_x + (end_x - start_x) * progress
            current_y = start_y + (end_y - start_y) * progress
            
            tile_color, text_color = get_tile_colors(val)
            pygame.draw.rect(screen, tile_color, (current_x, current_y, CELL_SIZE, CELL_SIZE))
            text_surface = TILE_FONT.render(str(val), True, text_color)
            text_rect = text_surface.get_rect(center=(current_x + CELL_SIZE / 2, 
                                                      current_y + CELL_SIZE / 2))
            screen.blit(text_surface, text_rect)

        elif anim['type'] == 'merge_moving_tile':
            r_from, c_from = anim['from_rc']
            r_to, c_to = anim['to_rc'] # Это куда она едет (место слияния)
            val = anim['value'] # значение движущейся плитки

            start_x, start_y = rc_to_pixels(r_from, c_from)
            end_x, end_y = rc_to_pixels(r_to, c_to)

            current_x = start_x + (end_x - start_x) * progress
            current_y = start_y + (end_y - start_y) * progress
            
            # Эта плитка исчезнет в конце, так что можно добавить эффект альфа-канала
            alpha = 255 * (1 - progress**2) # Быстрее исчезает
            
            tile_color, text_color = get_tile_colors(val)
            
            # Создаем временную поверхность для альфа-смешивания, если нужно
            temp_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(temp_surface, (*tile_color, int(alpha)), (0,0, CELL_SIZE, CELL_SIZE))
            text_s = TILE_FONT.render(str(val), True, text_color)
            text_s.set_alpha(int(alpha))
            text_r = text_s.get_rect(center=(CELL_SIZE/2, CELL_SIZE/2))
            temp_surface.blit(text_s, text_r)
            screen.blit(temp_surface, (current_x, current_y))

        elif anim['type'] == 'merge_target_tile':
            # Эта плитка "принимает" слияние. Она будет отображаться только в конце анимации
            # со своим новым значением, возможно, с эффектом пульсации.
            # Пока что, если анимация merge_moving_tile еще не закончена, эту плитку не рисуем (она скрыта под ней).
            # После завершения всех движений, она появится со значением merged_value.
            
            # Найдем соответствующую merge_moving_tile анимацию, чтобы знать, когда ее рисовать
            # Это упрощение - лучше иметь явную зависимость или тайминг
            related_moving_finished = True
            for other_anim in active_animations:
                if other_anim['type'] == 'merge_moving_tile' and other_anim['to_rc'] == anim['pos_rc']:
                    if other_anim['progress'] < 0.95:
                        related_moving_finished = False
                        break
            
            if related_moving_finished or progress > 0.1: # Начинаем пульсацию немного раньше или когда движение завершено
                r, c = anim['pos_rc']
                # Рисуем плитку с merged_value
                # Можно добавить эффект пульсации здесь, изменяя размер или цвет в зависимости от anim['progress']
                merged_val = anim['merged_value']
                tile_color, text_color = get_tile_colors(merged_val)
                
                current_size_factor = 1.0 + 0.2 * np.sin(progress * np.pi) # Плавная пульсация (0->0.2->0)
                pulse_size = CELL_SIZE * current_size_factor
                pulse_offset = (CELL_SIZE - pulse_size) / 2
                pulse_x, pulse_y = rc_to_pixels(r,c)

                pygame.draw.rect(screen, tile_color, (pulse_x + pulse_offset, pulse_y + pulse_offset, pulse_size, pulse_size))
                text_surface = TILE_FONT.render(str(merged_val), True, text_color)
                # Масштабирование текста - сложнее, пока оставим стандартный
                text_rect = text_surface.get_rect(center=(pulse_x + CELL_SIZE / 2, 
                                                          pulse_y + CELL_SIZE / 2))
                screen.blit(text_surface, text_rect)

def draw_game_over():
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_WIDTH), pygame.SRCALPHA) # SRFALPHA для прозрачности
    overlay.fill((238, 228, 218, 200)) # Полупрозрачный белый фон поверх доски
    screen.blit(overlay, (0,0))

    game_over_text = GAME_OVER_FONT.render("Game Over!", True, GAME_OVER_FONT_COLOR)
    restart_text = SCORE_LABEL_FONT.render("Press R to Restart or Q to Quit", True, GAME_OVER_FONT_COLOR)
    
    text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_WIDTH / 2 - 30))
    restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_WIDTH / 2 + 30))

    screen.blit(game_over_text, text_rect)
    screen.blit(restart_text, restart_rect)

def main():
    global ai_active, ai_agent, ai_model_loaded, torch
    game = Game()
    clock = pygame.time.Clock()
    running = True
    game_over_state = False
    ai_last_move_time = 0 # Таймер для задержки ходов ИИ
    log_file = None # Инициализируем как None
    debug_folder = "debug"
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(debug_folder, f"ai_debug_log_{timestamp}.txt")
    
    try:
        log_file = open(log_filename, 'w', encoding='utf-8')
        print(f"Отладочный лог будет сохранен в: {log_filename}")

        def log_debug_info(message):
            print(message)
            if log_file:
                log_file.write(message + '\n')

        while running:
            current_time = pygame.time.get_ticks()
            human_made_move_this_frame = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    # Если это событие QUIT, выходим из цикла по событиям и переходим к следующей итерации главного цикла
                    # или позволяем finally блоку закрыть файл лога, если running стало False.
                    # Нет нужды обрабатывать QUIT дальше.
            
                if event.type == pygame.KEYDOWN:
                    # Сначала обработаем общие клавиши управления (Q, M)
                    if event.key == pygame.K_q:
                        running = False
                        # Прерываем обработку текущего события, так как это выход
                        continue 
                    
                    if event.key == pygame.K_m:
                        ai_active = not ai_active
                        if ai_active and not ai_model_loaded:
                            log_debug_info("Активация ИИ. Загрузка модели...")
                            initialize_ai_agent() 
                            if not ai_model_loaded:
                                log_debug_info("Не удалось загрузить модель ИИ.")
                                ai_active = False
                            else:
                                ai_last_move_time = current_time 
                        else:
                            log_debug_info(f"ИИ {'активирован' if ai_active else 'деактивирован'}.")
                        # Прерываем обработку текущего события, так как это переключение режима
                        continue

                    # Теперь обрабатываем клавиши, зависящие от состояния игры
                    if game_over_state:
                        if event.key == pygame.K_r:
                            game = Game()
                            game_over_state = False
                            # При перезапуске стоит сбросить состояние ИИ, если он был активен
                            # ai_active = False # Решите, нужно ли сбрасывать ai_active при рестарте
                            # ai_model_loaded = False # Возможно, модель не нужно перезагружать
                            active_animations.clear()
                            ai_last_move_time = 0 
                            log_debug_info("Игра перезапущена.")
                    else: # Не game_over_state
                        # Ходы человека обрабатываются только если:
                        # - не конец игры
                        # - ИИ не активен
                        # - нет активных анимаций
                        if not ai_active and not active_animations:
                            animation_events_from_game = []
                            moved = False
                            if event.key == pygame.K_UP:
                                animation_events_from_game = game.move(0)
                                moved = True
                            elif event.key == pygame.K_DOWN:
                                animation_events_from_game = game.move(1)
                                moved = True
                            elif event.key == pygame.K_LEFT:
                                animation_events_from_game = game.move(2)
                                moved = True
                            elif event.key == pygame.K_RIGHT:
                                animation_events_from_game = game.move(3)
                                moved = True
                            
                            if moved and animation_events_from_game: # Убедимся, что ход был и он что-то изменил
                                human_made_move_this_frame = True
                                for ev in animation_events_from_game:
                                    add_animation_from_event(ev)
                            
                            if moved and game.is_game_over(): # Проверяем game over только если был сделан ход
                                game_over_state = True
                                log_debug_info("Game Over (после хода человека).")
                # Конец блока if event.type == pygame.KEYDOWN
        
            # --- Логика хода ИИ --- 
            if ai_active and ai_model_loaded and not game_over_state and not active_animations and not human_made_move_this_frame:
                if current_time - ai_last_move_time >= AI_MOVE_DELAY_MS:
                    current_game_state_for_ai = get_state(game.board)
                    
                    # --- Отладочный вывод перед ходом ИИ ---
                    log_debug_info("\n--- AI Making a Move ---")
                    log_debug_info("Board BEFORE AI move:")
                    for row_idx, game_row in enumerate(game.board): # Используем game_row вместо row
                        log_debug_info(str(game_row)) # Преобразуем в строку для записи в файл
                    log_debug_info(f"Score BEFORE AI move: {game.score}")
                    # --- Конец отладочного вывода ---

                    # Сохраняем состояние ДО хода для расчета вознаграждения
                    board_before_ai_move_raw = [list(r) for r in game.board] # Глубокая копия
                    score_before_ai_move = game.score

                    if ai_agent: 
                        ai_action = ai_agent.act(current_game_state_for_ai)
                        
                        # --- Отладочный вывод: выбранное действие ---
                        action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
                        log_debug_info(f"AI Action Chosen: {ai_action} ({action_map.get(ai_action, 'UNKNOWN')})")
                        # --- Конец отладочного вывода ---

                        animation_events_from_game = game.move(ai_action)
                        
                        board_changed_by_ai_move = bool(animation_events_from_game)
                        game_over_after_ai_move = game.is_game_over()
                        score_after_ai_move = game.score
                        board_after_ai_move_raw = game.board # Это уже состояние ПОСЛЕ хода

                        # Рассчитываем и выводим вознаграждение за этот ход
                        debug_reward = calculate_reward_for_debug(
                            score_before_ai_move, score_after_ai_move,
                            board_before_ai_move_raw, board_after_ai_move_raw,
                            board_changed_by_ai_move, game_over_after_ai_move
                        )
                        log_debug_info(f"Board AFTER AI move (Score: {score_after_ai_move}):")
                        for row_idx, game_row in enumerate(board_after_ai_move_raw): # game_row вместо row
                            log_debug_info(str(game_row)) # Преобразуем в строку для записи в файл
                        log_debug_info(f"DEBUG REWARD for AI's move: {debug_reward:.4f}")
                        if board_changed_by_ai_move:
                            log_debug_info("Board was changed by AI move.")
                        else:
                            log_debug_info("Board was NOT changed by AI move.")
                        if game_over_after_ai_move:
                            log_debug_info("GAME OVER after AI move.")
                        log_debug_info("--- End AI Move Debug ---")
                        # --- Конец отладочного вывода ---


                        if animation_events_from_game:
                            for ev in animation_events_from_game:
                                add_animation_from_event(ev)
                        # Обновляем таймер после КАЖДОЙ попытки хода ИИ, даже если ход ничего не изменил
                        # чтобы была задержка перед следующей попыткой.
                        ai_last_move_time = current_time 
                        if game.is_game_over(): game_over_state = True
            
            # Обновление анимаций
            if active_animations:
                for anim in list(active_animations): # Итерация по копии списка
                    elapsed_time = current_time - anim['start_time']
                    anim['progress'] = min(elapsed_time / ANIMATION_DURATION_MS, 1.0)
                    if anim['progress'] >= 1.0:
                        active_animations.remove(anim)

            if not game_over_state:
                draw_board(game.board, game.score)
            else:
                draw_board(game.board, game.score) # Показываем финальную доску
                draw_game_over()

            pygame.display.flip() # Обновляем весь экран
            clock.tick(60) 
    finally:
        if log_file:
            log_debug_info(f"Завершение игры. Лог сохранен в {log_filename}")
            log_file.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 