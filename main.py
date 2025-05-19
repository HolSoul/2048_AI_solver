import pygame
import sys
import os # Для проверки существования файла модели
import numpy as np # Для get_state
from game import Game
# from ai_solver import AISolver # Старый placeholder, если был
from ai_solver import DQNAgent, DEVICE, QNetwork # Наш агент и устройство, импортируем также QNetwork и torch из ai_solver, если они там

# Попытка импортировать torch глобально в main.py для проверки в initialize_ai_agent
if 'torch' not in globals(): # Если еще не импортирован (например, из ai_solver)
    try:
        import torch
        print("(main.py) PyTorch импортирован для проверки.")
    except ImportError:
        print("(main.py) PyTorch не найден. ИИ не будет доступен.")
        torch = None 

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
DEFAULT_MODEL_FILENAME = "dqn_2048_pytorch_ep1600.pth" # Или конкретный файл, например dqn_2048_pytorch_ep2000.pth
AI_MOVE_DELAY_MS = 100 # Задержка между ходами ИИ в миллисекундах (0.5 секунды)

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

    while running:
        current_time = pygame.time.get_ticks()
        human_made_move_this_frame = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False 
                if event.key == pygame.K_m:
                    ai_active = not ai_active
                    if ai_active and not ai_model_loaded:
                        print("Активация ИИ. Загрузка модели...")
                        initialize_ai_agent() 
                        if not ai_model_loaded:
                            print("Не удалось загрузить модель ИИ.")
                            ai_active = False
                        else:
                            ai_last_move_time = current_time # Сброс таймера при активации ИИ, чтобы он не ходил сразу
                    else:
                        print(f"ИИ {'активирован' if ai_active else 'деактивирован'}.")
                
                if game_over_state:
                    if event.key == pygame.K_r:
                        game = Game()
                        game_over_state = False
                        ai_active = False 
                        ai_model_loaded = False 
                        active_animations.clear()
                        ai_last_move_time = 0 # Сброс таймера
                elif not ai_active and not active_animations: 
                    animation_events_from_game = []
                    if event.key == pygame.K_UP: animation_events_from_game = game.move(0)
                    elif event.key == pygame.K_DOWN: animation_events_from_game = game.move(1)
                    elif event.key == pygame.K_LEFT: animation_events_from_game = game.move(2)
                    elif event.key == pygame.K_RIGHT: animation_events_from_game = game.move(3)
                    
                    if animation_events_from_game:
                        human_made_move_this_frame = True
                        for ev in animation_events_from_game:
                            add_animation_from_event(ev)
                    if game.is_game_over(): game_over_state = True
        
        # --- Логика хода ИИ --- 
        if ai_active and ai_model_loaded and not game_over_state and not active_animations and not human_made_move_this_frame:
            if current_time - ai_last_move_time >= AI_MOVE_DELAY_MS:
                current_game_state_for_ai = get_state(game.board)
                if ai_agent: 
                    ai_action = ai_agent.act(current_game_state_for_ai)
                    animation_events_from_game = game.move(ai_action)
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

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 