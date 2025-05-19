import pygame
import sys
from game import Game
from ai_solver import AISolver

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
GRID_COLOR = (205, 193, 180)
FONT_COLOR = (119, 110, 101)
SCORE_FONT_COLOR = (238, 228, 218)
GAME_OVER_FONT_COLOR = (119, 110, 101)


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
except pygame.error as e:
    print(f"Ошибка загрузки шрифта Arial: {e}. Используется шрифт по умолчанию.")
    TILE_FONT = pygame.font.Font(None, 55) # Шрифт по умолчанию, если Arial недоступен
    SCORE_LABEL_FONT = pygame.font.Font(None, 30)
    SCORE_FONT = pygame.font.Font(None, 35)
    GAME_OVER_FONT = pygame.font.Font(None, 60)


# Настройка экрана
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2048 Game")

# Анимации
ANIMATION_DURATION_MS = 150 # длительность анимации в миллисекундах
active_animations = [] # Теперь будет содержать более сложные объекты анимации

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
        # Мы создадим два объекта анимации или один сложный.
        # Пока что, упростим: одна плитка движется, другая меняет значение в конце.
        # Основная анимация будет для движущейся плитки from_rc2.
        # Плитка from_rc1 (которая становится to_rc) будет "скрыта" во время движения from_rc2 к ней,
        # а затем появится с новым значением (возможно, с пульсацией позже).
        
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
            alpha = 255 * (1 - progress) if progress > 0.7 else 255 
            
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
                    if other_anim['progress'] < 1.0:
                        related_moving_finished = False
                        break
            
            if related_moving_finished or progress >= 0.9: # Рисуем ее, когда движение почти/завершено
                                                        # или когда ее собственная анимация "пульсации" активна
                r, c = anim['pos_rc']
                # Рисуем плитку с merged_value
                # Можно добавить эффект пульсации здесь, изменяя размер или цвет в зависимости от anim['progress']
                merged_val = anim['merged_value']
                tile_color, text_color = get_tile_colors(merged_val)
                
                current_size_factor = 1.0
                if progress < 0.5 : # первая половина - увеличение
                    current_size_factor = 1.0 + 0.2 * (progress * 2) # Увеличение до 1.2
                elif progress < 1.0: # вторая половина - уменьшение обратно до 1.0
                    current_size_factor = 1.2 - 0.2 * ((progress - 0.5) * 2) # Уменьшение до 1.0
                
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
    game = Game() # Создаем экземпляр нашей игры из game.py
    clock = pygame.time.Clock()
    running = True
    game_over_state = False

    while running:
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if not active_animations: # Принимаем ввод, только если нет активных анимаций
                    if game_over_state:
                        if event.key == pygame.K_q:
                            running = False
                        if event.key == pygame.K_r:
                            game = Game() # Новая игра
                            game_over_state = False
                            active_animations.clear() # Очищаем анимации для новой игры
                    else:
                        animation_events_from_game = []
                        if event.key == pygame.K_UP:
                            animation_events_from_game = game.move(0) # 0: Up
                        elif event.key == pygame.K_DOWN:
                            animation_events_from_game = game.move(1) # 1: Down
                        elif event.key == pygame.K_LEFT:
                            animation_events_from_game = game.move(2) # 2: Left
                        elif event.key == pygame.K_RIGHT:
                            animation_events_from_game = game.move(3) # 3: Right
                        elif event.key == pygame.K_q: # Для выхода во время игры
                            running = False
                        
                        if animation_events_from_game:
                            for ev in animation_events_from_game:
                                add_animation_from_event(ev)
                        
                        if game.is_game_over():
                            game_over_state = True
        
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
        clock.tick(30) # Ограничение до 30 FPS

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 