import torch
from stable_baselines3 import PPO

MODEL_PATH = "ppo2_2048_custom_cnn.zip"
EXPORTED_MODEL_PATH = "ppo_2048_traced.pt"

# Загружаем модель
model = PPO.load(MODEL_PATH, device="cpu")

# Убедимся, что модель в режиме оценки
model.policy.eval()

# Пример входных данных (observation) для вашей модели
# Размерность должна соответствовать входу вашей CNN (например, 1x4x4)
dummy_input = torch.randn(1, 1, 4, 4) 

# Трассировка модели
traced_script_module = torch.jit.trace(model.policy, dummy_input)

# Сохранение трассированной модели
traced_script_module.save(EXPORTED_MODEL_PATH)

print(f"Модель успешно экспортирована в {EXPORTED_MODEL_PATH}")