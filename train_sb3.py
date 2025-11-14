# file: train_sb3.py

import json
from pathlib import Path

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from game_env import Game2048Env # Наша среда

# --- ШАГ 1: Создание кастомного CNN Feature Extractor ---
# Эта нейросеть будет специально разработана для обработки доски 4x4.
class CustomCnnExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: The observation space of the environment.
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # Мы предполагаем, что на вход приходит (N, 1, 4, 4)
        # N - размер батча, 1 - количество каналов, 4x4 - размер доски
        
        # Определяем нашу простую сверточную сеть
        self.cnn = nn.Sequential(
            # Первый сверточный слой: 1 входной канал, 32 выходных, ядро 2x2
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            # Второй сверточный слой: 32 входных, 64 выходных, ядро 2x2
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            # "Выпрямляем" результат в плоский вектор
            nn.Flatten(),
        )

        # Вычисляем размерность после сверток, чтобы создать правильный линейный слой
        # Для входа (1, 4, 4):
        # После Conv1(2x2): (32, 3, 3)
        # После Conv2(2x2): (64, 2, 2)
        # После Flatten: 64 * 2 * 2 = 256
        self.linear = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Применяем сверточные слои, затем линейный
        return self.linear(self.cnn(observations))


# --- Параметры ---
PRETRAINED_MODEL_PATH = Path("ppo2_2048_custom_cnn.zip")
MODEL_SAVE_PATH = "ppo2_2048_custom_cnn_finetuned.zip"
TOTAL_TIMESTEPS = 2_000_000
HARD_DATASET_PATH = Path("datasets/hard_positions.json")
HARD_START_PROB = 0.7  # Вероятность старта с хард позиции при каждом reset()

# Расчет оптимального размера датасета:
# При HARD_START_PROB=0.7 и средней длине игры ~750 шагов:
# - Количество ресетов = TOTAL_TIMESTEPS / 750
# - Хард стартов = ресетов * HARD_START_PROB
# - Использований на позицию = хард стартов / размер_датасета
# 
# Рекомендации для 120M timesteps:
# - 5,000-10,000 позиций: ~8-16 использований каждой позиции (хорошо)
# - 10,000-20,000 позиций: ~4-8 использований (отлично, больше разнообразия)
# - 1,000 позиций: ~112 использований (слишком часто, переобучение)


def load_hard_states():
    if HARD_DATASET_PATH.exists():
        with HARD_DATASET_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        dataset_size = len(data)
        print(f"Загружено сложных позиций: {dataset_size}")
        
        # Расчет статистики использования
        avg_game_length = 750  # примерная средняя длина игры
        num_resets = TOTAL_TIMESTEPS / avg_game_length
        hard_starts = num_resets * HARD_START_PROB
        uses_per_position = hard_starts / dataset_size if dataset_size > 0 else 0
        
        print(f"Ожидаемая статистика для {TOTAL_TIMESTEPS:,} timesteps:")
        print(f"  - Ресетов: ~{num_resets:,.0f}")
        print(f"  - Хард стартов: ~{hard_starts:,.0f} ({HARD_START_PROB*100:.0f}%)")
        print(f"  - Использований на позицию: ~{uses_per_position:.1f}")
        
        if uses_per_position > 50:
            print(f"  ⚠️  ВНИМАНИЕ: Каждая позиция будет использована >50 раз!")
            print(f"     Рекомендуется увеличить датасет до {int(hard_starts/20):,}-{int(hard_starts/10):,} позиций")
        elif uses_per_position > 20:
            print(f"  ⚠️  Каждая позиция будет использована >20 раз")
            print(f"     Рекомендуется увеличить датасет до {int(hard_starts/15):,}-{int(hard_starts/8):,} позиций")
        else:
            print(f"  ✓ Разнообразие датасета достаточное")
        
        return data
    print("Файл сложных позиций не найден, обучение начнется с обычных стартов.")
    return []


class MixedStartEnv(Game2048Env):
    def __init__(self, hard_states=None, hard_start_prob=0.5):
        super().__init__(state_mode='2d', reward_mode='complex')
        self.hard_states = hard_states or []
        self.hard_start_prob = hard_start_prob

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        if self.hard_states and self.np_random.random() < self.hard_start_prob:
            idx = int(self.np_random.integers(0, len(self.hard_states)))
            state = self.hard_states[idx]
            self.load_state(state["board"], state["score"])
            observation = self._get_obs()
            info = self._get_info()
        return observation, info


hard_states = load_hard_states()


def make_env():
    return MixedStartEnv(hard_states=hard_states, hard_start_prob=HARD_START_PROB)


# --- Настройка среды ---
env = DummyVecEnv([make_env])

# --- ШАГ 2: Обновление policy_kwargs для использования нашего Extractor ---
# Также исправляем предупреждение UserWarning, передавая net_arch как словарь, а не список
policy_kwargs = dict(
    features_extractor_class=CustomCnnExtractor,
    features_extractor_kwargs=dict(features_dim=128), # Размер выхода из нашего extractor
    net_arch=dict(pi=[128], vf=[128]) # MLP после extractor'а (policy и value сети)
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if PRETRAINED_MODEL_PATH.exists():
    print(f"Загружаем предобученную модель из {PRETRAINED_MODEL_PATH} для дообучения...")
    model = PPO.load(str(PRETRAINED_MODEL_PATH), env=env, device=device)
    model.policy.optimizer.param_groups[0]["lr"] = 5e-05
else:
    print("Предобученная модель не найдена, обучение начнется с нуля.")
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=9.320552204889768e-05,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.023494203671235357,
        verbose=1,
        device=device
    )

# --- Обучение ---
print(f"Начинаем обучение модели PPO с кастомной CNN на {TOTAL_TIMESTEPS} шагов...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

# --- Сохранение модели ---
model.save(MODEL_SAVE_PATH)
print(f"Обучение завершено. Модель сохранена в {MODEL_SAVE_PATH}")

# --- Проверка модели (опционально) ---
from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=20)
print(f"После обучения средняя награда за 20 игр: {mean_reward:.2f} +/- {std_reward:.2f}")