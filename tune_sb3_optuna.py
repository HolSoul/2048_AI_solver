# file: tune_sb3_optuna.py (Версия без warnings)

import optuna
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor # ИЗМЕНЕНО: Добавлен импорт Monitor

from game_env import Game2048Env

# --- Константы для подбора ---
N_TRIALS = 50
N_TIMESTEPS_TRIAL = 30_000
N_EVAL_EPISODES = 10
MAX_EPISODE_STEPS = 1000

# --- Кастомный CNN Extractor (без изменений) ---
class CustomCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0), nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(nn.Linear(256, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

def make_env():
    wrapped_env = Game2048Env(state_mode='2d', reward_mode='complex')
    wrapped_env = TimeLimit(wrapped_env, max_episode_steps=MAX_EPISODE_STEPS)
    return Monitor(wrapped_env)


def objective(trial: optuna.Trial) -> float:
    """Функция, которую Optuna будет оптимизировать."""
    
    env = make_env()
    
    # --- Гиперпараметры для подбора ---
    # ИЗМЕНЕНО: Используем suggest_float с log=True вместо suggest_loguniform
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
    gamma = trial.suggest_categorical('gamma', [0.99, 0.995, 0.999])
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True)
    clip_range = trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3])
    
    features_dim = trial.suggest_categorical('features_dim', [128, 256, 512])
    mlp_layer_size = trial.suggest_categorical('mlp_layer_size', [64, 128, 256])

    policy_kwargs = dict(
        features_extractor_class=CustomCnnExtractor,
        features_extractor_kwargs=dict(features_dim=features_dim),
        net_arch=dict(pi=[mlp_layer_size], vf=[mlp_layer_size])
    )

    print(f"[Trial {trial.number + 1}] Начало обучения: lr={learning_rate:.2e}, n_steps={n_steps}, "
          f"gamma={gamma}, ent_coef={ent_coef:.2e}, clip_range={clip_range}, "
          f"features_dim={features_dim}, mlp_layer_size={mlp_layer_size}")
    
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, learning_rate=learning_rate,
                n_steps=n_steps, gamma=gamma, gae_lambda=0.95, ent_coef=ent_coef, clip_range=clip_range,
                batch_size=64, n_epochs=10, verbose=0,
                device='cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model.learn(total_timesteps=N_TIMESTEPS_TRIAL)
    except (AssertionError, ValueError) as e:
        print(f"Ошибка в попытке {trial.number}: {e}")
        env.close()
        return -float('inf')
    
    # --- Оценка ---
    # ИЗМЕНЕНО: Оборачиваем среду в Monitor для корректной оценки
    eval_env = make_env()
    try:
        mean_reward, _ = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            warn=False
        )
        print(f"[Trial {trial.number + 1}] Средняя награда {mean_reward:.2f}")
    finally:
        env.close()
        eval_env.close()
    
    return mean_reward

# --- Запуск исследования ---
study = optuna.create_study(direction="maximize")
try:
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
except KeyboardInterrupt:
    print("Подбор прерван пользователем.")

# --- Результаты ---
print("\n========================================================")
print("Подбор гиперпараметров завершен!")
print(f"Количество завершенных попыток: {len(study.trials)}")
if study.best_trial:
    print("\nЛучшая попытка:")
    best = study.best_trial
    print(f"  > Значение (средняя награда): {best.value:.2f}")
    print("  > Лучшие гиперпараметры:")
    for key, value in best.params.items():
        print(f"    - {key}: {value}")
print("========================================================\n")