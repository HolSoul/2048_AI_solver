# file: game_env.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from game import Game  # Ваша оригинальная игровая логика

TARGET_CORNER_RC = (3, 0)  # Левый нижний угол (строка 3, столбец 0)

class Game2048Env(gym.Env):
    """
    Gymnasium-совместимая среда для игры 2048 с настраиваемым состоянием и наградой.
    
    Args:
        state_mode (str): 'flat' для 1D-вектора (16,) или '2d' для 2D-тензора (1, 4, 4).
        reward_mode (str): 'simple' (изменение счета), 'log_score' (лог. изменения), 'complex' (ваша эвристика).
    """
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, state_mode='flat', reward_mode='simple'):
        super().__init__()
        
        self.game = Game()
        self.size = self.game.size

        # Валидация режимов
        assert state_mode in ['flat', '2d'], "state_mode должен быть 'flat' или '2d'"
        assert reward_mode in ['simple', 'log_score', 'complex'], "reward_mode должен быть 'simple', 'log_score' или 'complex'"
        self.state_mode = state_mode
        self.reward_mode = reward_mode

        # Пространство действий: 0:Вверх, 1:Вниз, 2:Влево, 3:Вправо
        self.action_space = spaces.Discrete(4)

        # Пространство состояний (наблюдений)
        if self.state_mode == 'flat':
            # Плоский вектор 16x1, нормализованный
            self.observation_space = spaces.Box(low=0.0, high=1.0, 
                                                shape=(self.size * self.size,), 
                                                dtype=np.float32)
        else: # '2d'
            # 2D-тензор 1x4x4 для CNN (Каналы, Высота, Ширина)
            self.observation_space = spaces.Box(low=0.0, high=1.0, 
                                                shape=(1, self.size, self.size), 
                                                dtype=np.float32)

    def _get_obs(self):
        """Преобразует доску в нормализованное состояние в зависимости от state_mode."""
        # Нормализуем значения плиток логарифмом
        processed_board = np.zeros(shape=(self.size, self.size), dtype=np.float32)
        for r in range(self.size):
            for c in range(self.size):
                val = self.game.board[r][c]
                if val > 0:
                    processed_board[r, c] = np.log2(val) / 11.0  # Нормализация на log2(2048)

        if self.state_mode == 'flat':
            return processed_board.flatten()
        else: # '2d'
            return np.expand_dims(processed_board, axis=0) # Добавляем ось для канала -> (1, 4, 4)

    def _get_info(self):
        return {"score": self.game.score, "max_tile": np.max(self.game.board), "board": self.game.board}
    
    def _get_max_tile_value_and_loc(self, board_raw):
        max_val = 0
        loc = (-1, -1)
        if not board_raw:
            return 0, (-1, -1)
        board_np_flat = np.array(board_raw).flatten()
        if not np.any(board_np_flat):
            return 0, (-1, -1)
        for r_idx, row in enumerate(board_raw):
            for c_idx, val in enumerate(row):
                if val > max_val:
                    max_val = val
                    loc = (r_idx, c_idx)
        if max_val == 0:
            return 0, (-1, -1)
        return max_val, loc
    
    def _calculate_line_monotonicity_and_smoothness(self, line_array_raw):
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
                empty_at_end += 1
            else:
                break
        score += empty_at_end * 0.25
        return score

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game()
        return self._get_obs(), self._get_info()

    def step(self, action):
        score_before = self.game.score
        board_before_raw = [list(row) for row in self.game.board]

        animation_events = self.game.move(action)
        board_changed = bool(animation_events)

        reward = self._calculate_reward(
            score_before, self.game.score,
            board_before_raw, self.game.board,
            board_changed, self.game.is_game_over()
        )

        terminated = self.game.is_game_over()
        truncated = False
        observation = self._get_obs()
        info = self._get_info()
        info["animation_events"] = animation_events
        
        return observation, reward, terminated, truncated, info

    def render(self):
        print(f"Score: {self.game.score}")
        for row in self.game.board:
            print("\t".join(map(str, row)))
            
    def _calculate_reward(self, score_before, score_after, board_before, board_after, changed, is_over):
        """Рассчитывает награду в зависимости от выбранного reward_mode."""
        
        # --- Режим 1: Упрощенная награда ---
        if self.reward_mode == 'simple':
            if not changed and not is_over:
                return -1.0 # Небольшой штраф за невалидный ход
            return float(score_after - score_before) # Награда - это просто приросшие очки

        # --- Режим 2: Логарифмическая награда за очки ---
        elif self.reward_mode == 'log_score':
            if not changed and not is_over:
                return -1.0
            score_delta = score_after - score_before
            if score_delta > 0:
                return np.log2(score_delta)
            return 0.0

        # --- Режим 3: Ваша сложная эвристическая награда ---
        elif self.reward_mode == 'complex':
            reward = 0.0
            max_tile_before, loc_max_tile_before = self._get_max_tile_value_and_loc(board_before)
            max_tile_after, loc_max_tile_after = self._get_max_tile_value_and_loc(board_after)
            num_empty_after = self._count_empty_cells(board_after)
            board_np_after = np.array(board_after)
            board_np_before = np.array(board_before)
            
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
            current_target_corner = TARGET_CORNER_RC
            
            if not changed and not is_over:
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

            HIGH_TILE_ROW_BONUS_MULTIPLIER = 6.0
            HIGH_TILE_ROW_PENALTY_MULTIPLIER = 12.0
            HIGH_TILE_COLUMN_STACK_PENALTY = 8.0
            MIN_HIGH_TILE_VALUE = 32

            tiles_with_pos = []
            for r_idx in range(board_np_after.shape[0]):
                for c_idx in range(board_np_after.shape[1]):
                    val = board_np_after[r_idx, c_idx]
                    if val > 0:
                        tiles_with_pos.append((val, (r_idx, c_idx)))
            tiles_with_pos.sort(key=lambda x: x[0], reverse=True)

            bottom_row_index = current_target_corner[0]
            target_column_index = current_target_corner[1]

            for rank, (tile_val, (r_pos, c_pos)) in enumerate(tiles_with_pos[:6]):
                if tile_val < MIN_HIGH_TILE_VALUE:
                    break
                tile_log = np.log2(tile_val + epsilon_log)

                if r_pos != bottom_row_index and tile_val >= MIN_HIGH_TILE_VALUE * 2:
                    reward -= tile_log * HIGH_TILE_ROW_PENALTY_MULTIPLIER

                if r_pos == bottom_row_index and c_pos != target_column_index:
                    reward += tile_log * HIGH_TILE_ROW_BONUS_MULTIPLIER

                if c_pos == target_column_index and r_pos < bottom_row_index and tile_val >= MIN_HIGH_TILE_VALUE * 2:
                    reward -= tile_log * HIGH_TILE_COLUMN_STACK_PENALTY
            
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
                    reward += PENALTY_MAX_TILE_LEFT_CORNER
            
            grid_score = 0.0
            for r in range(board_np_after.shape[0]):
                row = board_np_after[r, :]
                weight = 1.0 if r == current_target_corner[0] else 0.7
                grid_score += self._calculate_line_monotonicity_and_smoothness(row) * weight
            for c_col_idx in range(board_np_after.shape[1]):
                col = board_np_after[:, c_col_idx]
                weight = 1.0 if c_col_idx == current_target_corner[1] else 0.7
                if current_target_corner[0] == 0:
                    grid_score += self._calculate_line_monotonicity_and_smoothness(col) * weight
                elif current_target_corner[0] == board_np_after.shape[0] - 1:
                    grid_score += self._calculate_line_monotonicity_and_smoothness(col[::-1]) * weight
            reward += grid_score * REWARD_MONOTONICITY_OVERALL_WEIGHT
            
            smoothness_penalty = 0.0
            for r_idx in range(board_np_after.shape[0]):
                for c_idx_smooth in range(board_np_after.shape[1]):
                    current_val = board_np_after[r_idx, c_idx_smooth]
                    if current_val == 0:
                        continue
                    if c_idx_smooth + 1 < board_np_after.shape[1]:
                        right_val = board_np_after[r_idx, c_idx_smooth+1]
                        if right_val != 0 and abs(np.log2(current_val+epsilon_log) - np.log2(right_val+epsilon_log)) > 1.0:
                            smoothness_penalty -= abs(np.log2(current_val+epsilon_log) - np.log2(right_val+epsilon_log))
                    if r_idx + 1 < board_np_after.shape[0]:
                        down_val = board_np_after[r_idx+1, c_idx_smooth]
                        if down_val != 0 and abs(np.log2(current_val+epsilon_log) - np.log2(down_val+epsilon_log)) > 1.0:
                            smoothness_penalty -= abs(np.log2(current_val+epsilon_log) - np.log2(down_val+epsilon_log))
            reward += smoothness_penalty * REWARD_SMOOTHNESS_WEIGHT
            
            potential_merges = 0
            for r_idx in range(board_np_after.shape[0]):
                for c_idx_pm in range(board_np_after.shape[1]):
                    current_val_pm = board_np_after[r_idx, c_idx_pm]
                    if current_val_pm == 0:
                        continue
                    if c_idx_pm + 1 < board_np_after.shape[1]:
                        if board_np_after[r_idx, c_idx_pm + 1] == current_val_pm:
                            potential_merges += 1
                    if r_idx + 1 < board_np_after.shape[0]:
                        if board_np_after[r_idx + 1, c_idx_pm] == current_val_pm:
                            potential_merges += 1
            reward += potential_merges * REWARD_POTENTIAL_MERGES_MULTIPLIER
            
            if is_over:
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
                    if max_tile_after < 32:
                        reward -= 100.0
                    elif max_tile_after < 128:
                        reward -= 50.0
            
            return np.clip(reward, -400.0, 1700.0)

    # --- Вспомогательные функции для 'complex' reward ---
    def _count_empty_cells(self, board_raw):
        return np.sum(np.array(board_raw) == 0)
    # ... (добавьте сюда другие вспомогательные функции, если они нужны для 'complex' награды)

    def load_state(self, board_state, score):
        board_array = np.array(board_state, dtype=np.int32)
        if board_array.shape != (self.size, self.size):
            raise ValueError("Некорректный размер доски при загрузке состояния")
        self.game.board = board_array.tolist()
        self.game.score = int(score)
        self.game.last_added_tile_info = None