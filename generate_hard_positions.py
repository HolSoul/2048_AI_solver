import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import TimeLimit

from game_env import Game2048Env
import pygame


MODEL_PATH = "ppo2_2048_custom_cnn.zip"
OUTPUT_PATH = Path("datasets") / "hard_positions.json"
NUM_ENVS = 10
TARGET_DATASET_SIZE = 100
MIN_MAX_TILE = 64
MAX_EMPTY_CELLS = 8
MIN_EMPTY_CELLS = 1
VISUALIZE = True

# Визуализация: сетка 5x4 для 20 окружений
GRID_COLS = 5
GRID_ROWS = 4
TILE_SIZE = 26
CELL_GAP = 2
BOARD_PAD = 4
FONT_SIZE = 14


def make_env():
    def _init():
        base = Game2048Env(state_mode="2d", reward_mode="complex")
        return TimeLimit(base, max_episode_steps=800)

    return _init


def is_hard_position(board: np.ndarray) -> bool:
    max_tile = board.max()
    empty_cells = np.sum(board == 0)
    if max_tile < MIN_MAX_TILE:
        return False
    return MIN_EMPTY_CELLS <= empty_cells <= MAX_EMPTY_CELLS


def main():
    output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])
    model = PPO.load(MODEL_PATH, env=env, device="cpu")

    # Визуализация
    screen = None
    font = None
    running_vis = VISUALIZE
    if running_vis:
        pygame.init()
        board_pix = BOARD_PAD * 2 + 4 * TILE_SIZE + 3 * CELL_GAP
        screen_w = GRID_COLS * board_pix + (GRID_COLS - 1) * BOARD_PAD
        progress_h = 28
        screen_h = GRID_ROWS * board_pix + (GRID_ROWS - 1) * BOARD_PAD + progress_h
        screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption("Collecting hard 2048 positions")
        font = pygame.font.SysFont("Arial", FONT_SIZE)

    def tile_color(v: int) -> Tuple[int, int, int]:
        if v == 0:
            return (205, 193, 180)
        level = int(np.log2(v)) if v > 0 else 0
        palette = {
            1: (238, 228, 218),
            2: (237, 224, 200),
            3: (242, 177, 121),
            4: (245, 149, 99),
            5: (246, 124, 95),
            6: (246, 94, 59),
            7: (237, 207, 114),
            8: (237, 204, 97),
            9: (237, 200, 80),
            10: (237, 197, 63),
            11: (237, 194, 46),
        }
        return palette.get(level, (60, 58, 50))

    def draw(infos_list: List[Dict], collected: int):
        if not running_vis or screen is None:
            return
        screen.fill((250, 248, 239))

        board_pix = BOARD_PAD * 2 + 4 * TILE_SIZE + 3 * CELL_GAP
        for idx, info in enumerate(infos_list[:NUM_ENVS]):
            r_grid = idx // GRID_COLS
            c_grid = idx % GRID_COLS
            x0 = c_grid * (board_pix + BOARD_PAD)
            y0 = r_grid * (board_pix + BOARD_PAD)
            # фон доски
            pygame.draw.rect(screen, (187, 173, 160), (x0, y0, board_pix, board_pix), border_radius=6)

            board = info.get("board")
            if board is None:
                continue
            for r in range(4):
                for c in range(4):
                    val = int(board[r][c])
                    cx = x0 + BOARD_PAD + c * (TILE_SIZE + CELL_GAP)
                    cy = y0 + BOARD_PAD + r * (TILE_SIZE + CELL_GAP)
                    pygame.draw.rect(screen, tile_color(val), (cx, cy, TILE_SIZE, TILE_SIZE), border_radius=4)
                    if val > 0:
                        text = font.render(str(val), True, (119, 110, 101))
                        tr = text.get_rect(center=(cx + TILE_SIZE // 2, cy + TILE_SIZE // 2))
                        screen.blit(text, tr)

        # Прогресс
        progress_text = f"Collected: {collected}/{TARGET_DATASET_SIZE}"
        text = font.render(progress_text, True, (80, 75, 70))
        screen.blit(text, (8, GRID_ROWS * (board_pix + BOARD_PAD)))
        pygame.display.flip()

    observations = env.reset()
    collected_states: Dict[Tuple[int, ...], Dict] = {}
    printed_count = 0
    step_counter = 0

    # динамические пороги
    dynamic_min_max = MIN_MAX_TILE
    dynamic_max_empty = MAX_EMPTY_CELLS

    while len(collected_states) < TARGET_DATASET_SIZE:
        step_counter += 1
        if running_vis:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running_vis = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running_vis = False

        # небольшая доля случайных действий для лучшего покрытия состояний
        if (step_counter % 50) == 0:
            actions = np.random.randint(0, 4, size=(NUM_ENVS,), dtype=np.int64)
        else:
            actions, _ = model.predict(observations, deterministic=True)
        observations, _, dones, infos = env.step(actions)

        if running_vis:
            draw(infos, len(collected_states))

        for info in infos:
            board = np.array(info["board"], dtype=np.int32)
            # ужесточаем критерии после половины датасета
            if len(collected_states) >= TARGET_DATASET_SIZE // 2:
                dynamic_min_max = 128
                dynamic_max_empty = 6

            max_tile = int(board.max())
            empty_cells = int(np.sum(board == 0))
            if not (max_tile >= dynamic_min_max and MIN_EMPTY_CELLS <= empty_cells <= dynamic_max_empty):
                continue

            key = tuple(board.flatten().tolist())
            if key in collected_states:
                continue

            collected_states[key] = {
                "board": board.tolist(),
                "score": int(info["score"]),
                "max_tile": int(board.max()),
                "empty_cells": int(np.sum(board == 0)),
            }
            if len(collected_states) != printed_count:
                printed_count = len(collected_states)
                print(f"[collect] {printed_count}/{TARGET_DATASET_SIZE} "
                      f"(max={max_tile}, empty={empty_cells}, "
                      f"min_req={dynamic_min_max}, max_empty={dynamic_max_empty})", flush=True)

        if np.all(dones):
            observations = env.reset()
        elif step_counter % 200 == 0 and len(collected_states) == printed_count:
            # Heartbeat, если долго нет новых позиций
            print(f"[progress] steps={step_counter}, collected={printed_count}/{TARGET_DATASET_SIZE}", flush=True)

    dataset: List[Dict] = list(collected_states.values())[:TARGET_DATASET_SIZE]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    env.close()
    if running_vis:
        pygame.quit()


if __name__ == "__main__":
    main()

