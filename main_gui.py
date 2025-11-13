# file: main_gui.py

import pygame
import sys
import os
import numpy as np
import torch

# Импортируем нашу новую среду и агентов
from game_env import Game2048Env
from ai_solver import DQNAgent # Для загрузки старого агента
from stable_baselines3 import PPO # Для загрузки нового агента

# --- ВАШ КОД: ИНИЦИАЛИЗАЦИЯ PYGAME, КОНСТАНТЫ, ЦВЕТА И ШРИФТЫ ---
# (Скопировано из вашего оригинального main.py)
# ===================================================================
pygame.init()
pygame.font.init()

# Константы
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 500
GRID_SIZE = 4
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE
GRID_LINE_WIDTH = 6
BACKGROUND_COLOR = (187, 173, 160)
FONT_COLOR = (119, 110, 101)
SCORE_FONT_COLOR = (238, 228, 218)
GAME_OVER_FONT_COLOR = (119, 110, 101)
AI_TEXT_COLOR = (255, 0, 0)

# Цвета для плиток
TILE_COLORS = {
    0: ((205, 193, 180), FONT_COLOR), 2: ((238, 228, 218), FONT_COLOR),
    4: ((237, 224, 200), FONT_COLOR), 8: ((242, 177, 121), SCORE_FONT_COLOR),
    16: ((245, 149, 99), SCORE_FONT_COLOR), 32: ((246, 124, 95), SCORE_FONT_COLOR),
    64: ((246, 94, 59), SCORE_FONT_COLOR), 128: ((237, 207, 114), SCORE_FONT_COLOR),
    256: ((237, 204, 97), SCORE_FONT_COLOR), 512: ((237, 200, 80), SCORE_FONT_COLOR),
    1024: ((237, 197, 63), SCORE_FONT_COLOR), 2048: ((237, 194, 46), SCORE_FONT_COLOR),
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
    TILE_FONT, SCORE_LABEL_FONT, SCORE_FONT, GAME_OVER_FONT, AI_STATUS_FONT = [pygame.font.Font(None, s) for s in [55, 30, 35, 60, 25]]

# Настройка экрана и глобальных переменных
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2048 Game with AI")
active_animations = []
ANIMATION_DURATION_MS = 150
# ===================================================================


# --- ВАШ КОД: ФУНКЦИИ ОТРИСОВКИ И АНИМАЦИИ ---
# (Скопировано из вашего оригинального main.py)
# ===================================================================
def rc_to_pixels(r, c):
    return c * CELL_SIZE, r * CELL_SIZE

def add_animation_from_event(event):
    now = pygame.time.get_ticks()
    anim_obj = {'type': event['type'], 'start_time': now, 'progress': 0}
    if event['type'] in ['appear', 'move']:
        anim_obj.update(event)
    elif event['type'] == 'merge':
        active_animations.append({
            'type': 'merge_moving_tile', 'start_time': now, 'progress': 0,
            'from_rc': event['from_rc2'], 'to_rc': event['to_rc'], 'value': event['original_value']
        })
        active_animations.append({
            'type': 'merge_target_tile', 'start_time': now, 'progress': 0,
            'pos_rc': event['to_rc'], 'original_value': event['original_value'],
            'merged_value': event['merged_value'], 'from_rc1_original_pos': event['from_rc1']
        })
        return
    active_animations.append(anim_obj)

def get_tile_colors(value):
    return TILE_COLORS.get(value, TILE_COLORS[2048])

def draw_board(board_data, current_score):
    screen.fill(BACKGROUND_COLOR)
    score_label = SCORE_LABEL_FONT.render("SCORE", True, FONT_COLOR)
    score_value = SCORE_FONT.render(str(current_score), True, SCORE_FONT_COLOR)
    screen.blit(score_label, (20, SCREEN_WIDTH + 10))
    screen.blit(score_value, (20, SCREEN_WIDTH + 35))

    status_text = "Human Player (M for AI)"
    status_color = FONT_COLOR
    if ai_active:
        status_text = "AI ACTIVE (M to toggle)" if ai_model_loaded else "AI FAILED (M to toggle)"
        status_color = AI_TEXT_COLOR
    ai_status = AI_STATUS_FONT.render(status_text, True, status_color)
    screen.blit(ai_status, (SCREEN_WIDTH - ai_status.get_width() - 10, SCREEN_WIDTH + 15))

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            is_involved = any(
                (anim.get('from_rc') == (r,c) or anim.get('to_rc') == (r,c) or anim.get('pos_rc') == (r,c) or anim.get('from_rc1_original_pos') == (r,c))
                for anim in active_animations
            )
            rect_x, rect_y = c * CELL_SIZE, r * CELL_SIZE
            if is_involved:
                pygame.draw.rect(screen, TILE_COLORS[0][0], (rect_x, rect_y, CELL_SIZE, CELL_SIZE))
                continue
            
            tile_value = board_data[r][c]
            tile_color, text_color = get_tile_colors(tile_value)
            pygame.draw.rect(screen, tile_color, (rect_x, rect_y, CELL_SIZE, CELL_SIZE))
            if c > 0: pygame.draw.line(screen, BACKGROUND_COLOR, (rect_x, rect_y), (rect_x, rect_y + CELL_SIZE), GRID_LINE_WIDTH)
            if r > 0: pygame.draw.line(screen, BACKGROUND_COLOR, (rect_x, rect_y), (rect_x + CELL_SIZE, rect_y), GRID_LINE_WIDTH)
            if tile_value != 0:
                text_surface = TILE_FONT.render(str(tile_value), True, text_color)
                text_rect = text_surface.get_rect(center=(rect_x + CELL_SIZE / 2, rect_y + CELL_SIZE / 2))
                screen.blit(text_surface, text_rect)

    for anim in list(active_animations):
        progress = anim['progress']
        if anim['type'] == 'appear':
            r, c = anim['pos_rc']; value = anim['value']
            color, text_color = get_tile_colors(value)
            size = CELL_SIZE * progress; offset = (CELL_SIZE - size) / 2
            x, y = c * CELL_SIZE + offset, r * CELL_SIZE + offset
            pygame.draw.rect(screen, color, (x, y, size, size))
            if progress > 0.7:
                text_surface = TILE_FONT.render(str(value), True, text_color)
                text_rect = text_surface.get_rect(center=(x - offset + CELL_SIZE / 2, y - offset + CELL_SIZE / 2))
                screen.blit(text_surface, text_rect)
        # (Остальная логика анимации скопирована без изменений)
        elif anim['type'] in ['move', 'merge_moving_tile']:
            r_f, c_f = anim['from_rc']; r_t, c_t = anim['to_rc']; val = anim['value']
            x_s, y_s = rc_to_pixels(r_f, c_f); x_e, y_e = rc_to_pixels(r_t, c_t)
            x, y = x_s + (x_e - x_s) * progress, y_s + (y_e - y_s) * progress
            color, text_color = get_tile_colors(val)
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
            text_surf = TILE_FONT.render(str(val), True, text_color)
            text_rect = text_surf.get_rect(center=(x + CELL_SIZE/2, y + CELL_SIZE/2))
            screen.blit(text_surf, text_rect)
        elif anim['type'] == 'merge_target_tile':
             if progress > 0.1:
                r, c = anim['pos_rc']; merged_val = anim['merged_value']
                color, text_color = get_tile_colors(merged_val)
                size_factor = 1.0 + 0.2 * np.sin(progress * np.pi)
                size = CELL_SIZE * size_factor; offset = (CELL_SIZE - size) / 2
                x, y = rc_to_pixels(r, c)
                pygame.draw.rect(screen, color, (x + offset, y + offset, size, size))
                text_surf = TILE_FONT.render(str(merged_val), True, text_color)
                text_rect = text_surf.get_rect(center=(x + CELL_SIZE/2, y + CELL_SIZE/2))
                screen.blit(text_surf, text_rect)

def draw_game_over():
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_WIDTH), pygame.SRCALPHA)
    overlay.fill((238, 228, 218, 200))
    screen.blit(overlay, (0, 0))
    game_over_text = GAME_OVER_FONT.render("Game Over!", True, GAME_OVER_FONT_COLOR)
    restart_text = SCORE_LABEL_FONT.render("Press R to Restart", True, GAME_OVER_FONT_COLOR)
    text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_WIDTH / 2 - 30))
    restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_WIDTH / 2 + 30))
    screen.blit(game_over_text, text_rect)
    screen.blit(restart_text, restart_rect)
# ===================================================================
# --- ИСПРАВЛЕННАЯ ЛОГИКА ЗАГРУЗКИ И УПРАВЛЕНИЯ ИИ ---
ai_agent_ranker = None # Новая переменная для функции-ранжировщика
ai_active = False
ai_model_loaded = False
AI_MOVE_DELAY_MS = 100

def load_ai_agent(filename):
    """
    Универсальная функция, которая загружает модель и возвращает
    функцию, выдающую ОТСОРТИРОВАННЫЙ список лучших действий.
    """
    global ai_agent_ranker, ai_model_loaded
    
    model_path_zip = f"./{filename}"
    model_path_pth = f"./models_pytorch/{filename}" # Путь для вашего DQN
    actual_path = None

    if os.path.exists(model_path_zip): actual_path = model_path_zip
    elif os.path.exists(model_path_pth): actual_path = model_path_pth
    else: print(f"Ошибка: Файл модели '{filename}' не найден ни в одной из директорий."); ai_model_loaded = False; return

    state_mode = '2d' if 'cnn' in filename or 'ppo' in filename else 'flat'
    env = Game2048Env(state_mode=state_mode)

    try: # Пытаемся загрузить как модель SB3 (.zip)
        model = PPO.load(actual_path, device='cpu')
        def get_ranked_actions_sb3(state):
            obs = torch.as_tensor(np.expand_dims(state, axis=0)).to(model.device)
            # Получаем "сырые" выходы (логиты) из сети, они отражают предпочтения
            dist = model.policy.get_distribution(obs)
            action_logits = dist.distribution.logits
            # Сортируем действия по убыванию их логитов
            return torch.argsort(action_logits, descending=True).squeeze().tolist()
        
        ai_agent_ranker = get_ranked_actions_sb3
        ai_model_loaded = True
        print(f"Модель Stable Baselines3 '{filename}' успешно загружена.")
        
    except Exception:
        try: # Пытаемся загрузить как ваш DQN агент (.pth)
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            
            agent = DQNAgent(state_size, action_size)
            agent.load(filename)
            agent.policy_net.eval()
            
            def get_ranked_actions_dqn(state):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to("cpu")
                with torch.no_grad():
                    q_values = agent.policy_net(state_tensor)
                # Сортируем действия по убыванию их Q-значений
                return torch.argsort(q_values, descending=True).squeeze().tolist()

            ai_agent_ranker = get_ranked_actions_dqn
            ai_model_loaded = True
            print(f"DQN модель '{filename}' успешно загружена.")
        except Exception as e:
            print(f"Не удалось загрузить модель '{filename}'. Ошибка: {e}")
            ai_model_loaded = False

def main():
    global ai_active, ai_agent_ranker, ai_model_loaded
    
    MODEL_TO_TEST = "ppo2_2048_custom_cnn_finetuned.zip"

    state_mode_for_gui = '2d' if 'cnn' in MODEL_TO_TEST or 'ppo' in MODEL_TO_TEST else 'flat'
    env = Game2048Env(state_mode=state_mode_for_gui)
    state, info = env.reset()

    clock = pygame.time.Clock()
    running = True
    game_over_state = False
    ai_last_move_time = 0

    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q: running = False; continue
                if event.key == pygame.K_m:
                    ai_active = not ai_active
                    if ai_active and not ai_model_loaded: load_ai_agent(MODEL_TO_TEST)
                    continue
                if game_over_state:
                    if event.key == pygame.K_r:
                        state, info = env.reset(); game_over_state = False; active_animations.clear()
                elif not ai_active and not active_animations:
                    key_map = {pygame.K_UP: 0, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_RIGHT: 3}
                    if event.key in key_map:
                        action = key_map[event.key]
                        state, _, terminated, _, info = env.step(action)
                        game_over_state = terminated
                        if info.get("animation_events"):
                            for ev in info["animation_events"]: add_animation_from_event(ev)
        
        # --- ИСПРАВЛЕННЫЙ ЦИКЛ ПРИНЯТИЯ РЕШЕНИЙ ИИ ---
        if ai_active and ai_model_loaded and not game_over_state and not active_animations:
            if current_time - ai_last_move_time >= AI_MOVE_DELAY_MS:
                
                # 1. Получаем отсортированный список лучших действий
                ranked_actions = ai_agent_ranker(state)
                
                # 2. Ищем первый валидный ход в этом списке
                final_action = -1
                for action in ranked_actions:
                    if env.game.peek_move(action): # Используем новый метод!
                        final_action = action
                        break # Нашли валидный ход, выходим из цикла
                
                # 3. Выполняем ход, если он был найден
                if final_action != -1:
                    state, _, terminated, _, info = env.step(final_action)
                    game_over_state = terminated
                    if info.get("animation_events"):
                        for ev in info["animation_events"]: add_animation_from_event(ev)
                else:
                    # Если ни один из 4 ходов не валиден, значит игра окончена
                    game_over_state = True
                
                ai_last_move_time = current_time

        if active_animations:
            for anim in list(active_animations):
                elapsed = current_time - anim['start_time']
                anim['progress'] = min(elapsed / ANIMATION_DURATION_MS, 1.0)
                if anim['progress'] >= 1.0: active_animations.remove(anim)

        draw_board(info['board'], info['score'])
        if game_over_state:
            draw_game_over()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()