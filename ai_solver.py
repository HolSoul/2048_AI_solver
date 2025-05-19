import numpy as np
import random
from collections import deque
import os # Для создания директории моделей

# Попытка импортировать PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    print("PyTorch успешно импортирован.")
    # Определяем устройство (CPU или GPU, если доступно)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {DEVICE}")
except ImportError as e:
    print(f"Ошибка импорта PyTorch: {e}")
    print("Пожалуйста, убедитесь, что PyTorch установлен.")
    torch = None
    DEVICE = "cpu" # По умолчанию, если PyTorch не импортирован

# Нейронная сеть для Q-функции
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size1=64, hidden_size2=64):
        super(QNetwork, self).__init__()
        # В PyTorch state_size - это количество входных признаков
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 replay_buffer_size=10000, batch_size=64, tau=1e-3,
                 model_save_path="./models_pytorch/"):
        if torch is None:
            raise ImportError("PyTorch не доступен. DQNAgent не может быть создан.")

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau # Для мягкого обновления целевой сети
        self.model_save_path = model_save_path

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # Q-Networks
        self.policy_net = QNetwork(state_size, action_size).to(DEVICE)
        self.target_net = QNetwork(state_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Инициализация весов целевой сети
        self.target_net.eval() # Целевая сеть в режиме оценки (не обучается)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE) # (1, state_size)
        self.policy_net.eval() # Переключаем в режим оценки для предсказания
        with torch.no_grad(): # Не вычисляем градиенты во время предсказания
            action_values = self.policy_net(state_tensor)
        self.policy_net.train() # Возвращаем в режим обучения

        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return action_values.cpu().data.numpy().argmax()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.from_numpy(np.vstack([e[0] for e in minibatch if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e[1] for e in minibatch if e is not None])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e[2] for e in minibatch if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e[3] for e in minibatch if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e[4] for e in minibatch if e is not None]).astype(np.uint8)).float().to(DEVICE)

        # Получаем Q-значения для следующих состояний из целевой сети
        # .max(1) возвращает (values, indices) для каждого ряда, берем values ([0])
        q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Вычисляем Q-цели для текущих состояний: R + gamma * max_a Q_target(s', a')
        # Для конечных состояний (done=1) Q-цель это просто награда R
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Получаем ожидаемые Q-значения из основной сети для выбранных действий
        # .gather(1, actions) выбирает Q-значения для тех действий, которые были реально совершены
        q_expected = self.policy_net(states).gather(1, actions)

        # Вычисляем функцию потерь
        loss = F.mse_loss(q_expected, q_targets)
        
        # Оптимизация модели
        self.optimizer.zero_grad() # Обнуляем градиенты
        loss.backward() # Вычисляем градиенты
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1) # Опционально: ограничение градиентов
        self.optimizer.step() # Обновляем веса

        # Мягкое обновление целевой сети
        self._soft_update_target_net()

        # Уменьшаем эпсилон
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _soft_update_target_net(self):
        """Мягкое обновление весов целевой сети: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def load(self, filename="dqn_2048_pytorch.pth"):
        load_path = os.path.join(self.model_save_path, filename)
        if os.path.exists(load_path):
            self.policy_net.load_state_dict(torch.load(load_path, map_location=DEVICE))
            self.target_net.load_state_dict(self.policy_net.state_dict()) # Синхронизируем
            self.policy_net.eval() # Переводим в режим оценки после загрузки
            self.target_net.eval()
            print(f"Модель успешно загружена с {load_path}")
        else:
            print(f"Файл модели {load_path} не найден. Обучение начнется с нуля.")

    def save(self, filename="dqn_2048_pytorch.pth"):
        save_path = os.path.join(self.model_save_path, filename)
        torch.save(self.policy_net.state_dict(), save_path)
        print(f"Модель сохранена в {save_path}")

# Пример использования (пока не интегрировано с игрой)
if __name__ == '__main__':
    if torch is not None:
        state_size_example = 16 # Например, доска 4x4, представленная как плоский вектор
        action_size_example = 4   # 4 возможных хода
        
        agent = DQNAgent(state_size_example, action_size_example)
        
        # Пример одного шага взаимодействия
        example_state = np.random.rand(state_size_example) # Случайное состояние
        action_chosen = agent.act(example_state)
        print(f"Состояние: {example_state}")
        print(f"Выбранное действие: {action_chosen}")
        
        # agent.save() # Пример сохранения
        # agent.load() # Пример загрузки

        print("DQNAgent (PyTorch) создан. Для использования нужна интеграция с игрой и цикл обучения.")
    else:
        print("Невозможно запустить пример DQNAgent, так как PyTorch не найден.")

# # Пример использования
# if __name__ == '__main__':
#     if torch is not None:
#         state_size_example = 16 # Например, доска 4x4, представленная как плоский вектор
#         action_size_example = 4   # 4 возможных хода
        
#         agent = DQNAgent(state_size_example, action_size_example)
        
#         # Пример одного шага взаимодействия
#         example_state = np.random.rand(state_size_example) # Случайное состояние
#         action_chosen = agent.act(example_state)
#         print(f"Состояние: {example_state}")
#         print(f"Выбранное действие: {action_chosen}")
        
#         # Пример сохранения/загрузки (создаст файл в текущей директории)
#         # agent.save("./dqn_2048_test.weights.h5")
#         # agent.load("./dqn_2048_test.weights.h5")

#         # Дальнейшие шаги: интеграция с game.py, цикл обучения.
#         print("DQNAgent создан. Для использования нужна интеграция с игрой и цикл обучения.")
#     else:
#         print("Невозможно запустить пример DQNAgent, так как PyTorch не найден.") 