import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer


class Real2SimDoublePendulumEnv(gym.Env):
    """Кастомная среда для двухколенного маятника с учетом ограничений Sim2Real"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(self, render_mode=None):
        super().__init__()
        self.viewer = None
        
        # 1. Параметры Sim2Real (настройки железной части)
        self.dt = 0.005          # 200 Гц — частота опроса на ESP32
        self.latency_steps = 2   # Задержка в 2 шага (10 мс) на чтение датчиков и отправку ШИМ
        self.noise_std = 0.005   # Величина шума энкодера AS5600 (~0.3 градуса)
        self.max_force = 15.0    # Ограничение по моменту мотора (Ньютоны)
        
        # Загружаем физическую модель MuJoCo (XML-строка или путь к файлу)
        # Для примера используем встроенный парсер. Модель содержит cart, pole1, pole2.
        self.model = mujoco.MjModel.from_xml_path("double_inverted_pendulum.xml")
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.dt
        
        self.render_mode = render_mode
        
        # 2. Пространство действий (Action Space): Управление мотором тележки [-1.0 ... 1.0]
        # Мы нормализуем действие, ИИ выдает значение от -1 до 1, мы умножаем его на max_force
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 3. Пространство состояний (Observation Space)
        # [позиция_тележки, sin(угол1), cos(угол1), sin(угол2), cos(угол2), скорость_тележки, скор_угл1, скор_угл2]
        # Использование sin/cos вместо «голых» углов исключает разрыв при переходе через 360 градусов.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        # Буфер для эмуляции задержки сигнала
        self.action_buffer = []

    def _get_obs(self):
        """Получение данных из физического движка + наложение шума реальных энкодеров"""
        # Извлекаем чистые данные из MuJoCo
        cart_pos = self.data.joint("slider").qpos[0]
        theta1   = self.data.joint("hinge1").qpos[0]
        theta2   = self.data.joint("hinge2").qpos[0]
        
        cart_vel   = self.data.joint("slider").qvel[0]
        theta1_dot = self.data.joint("hinge1").qvel[0]
        theta2_dot = self.data.joint("hinge2").qvel[0]
        
        # Наложение шума энкодера AS5600
        cart_pos += np.random.normal(0, self.noise_std)
        theta1   += np.random.normal(0, self.noise_std)
        theta2   += np.random.normal(0, self.noise_std)
        
        return np.array([
            cart_pos,
            np.sin(theta1), np.cos(theta1),
            np.sin(theta2), np.cos(theta2),
            cart_vel, theta1_dot, theta2_dot
        ], dtype=np.float32)

    def step(self, action):
        # 1. Применяем задержку сигнала (Action Latency)
        self.action_buffer.append(action)
        if len(self.action_buffer) > self.latency_steps:
            delayed_action = self.action_buffer.pop(0)
        else:
            delayed_action = np.zeros_like(action) # Пока буфер не полон, мотор стоит
            
        # 2. Масштабируем действие под возможности реального мотора (Actuator Saturation)
        u = delayed_action[0] * self.max_force
        self.data.ctrl[0] = u
        
        # 3. Делаем шаг физического симулятора
        mujoco.mj_step(self.model, self.data)
        
        # 4. Считаем наблюдения
    
        obs = self._get_obs()
        
        # СТРОГО фиксируем индексы из вектора obs:
        # obs = [cart_pos, sin_t1, cos_t1, sin_t2, cos_t2, cart_vel, t1_dot, t2_dot]
        cart_pos = obs[0]
        cos_t1   = obs[2]
        cos_t2   = obs[4]
        
        r_pose = 1.0 * cos_t1 + 2.0 * cos_t2
        r_control = -0.01 * np.square(action)[0]
        r_bounds = -0.5 * np.square(cart_pos)
        
        reward = r_pose + r_control + r_bounds
        
        terminated = bool(abs(cart_pos) > 0.8)
        if terminated:
            reward -= 100.0  
            
        if not terminated and cos_t1 > 0.95 and cos_t2 > 0.95:
            reward += 10.0
            
        return obs, reward, terminated, False, {}



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Полностью сбрасываем состояние модели и данные
        mujoco.mj_resetData(self.model, self.data)
        
        # 1. Задаем начальные ПОЗИЦИИ (qpos)
        self.data.qpos[0] = 0.0  # Тележка строго по центру рельса
        # Первое колено смотрит вниз (pi) + микро-отклонение, чтобы ИИ выходил из идеального баланса
        self.data.qpos[1] = np.pi + np.random.uniform(-0.05, 0.05) 
        # Второе колено смотрит вниз (pi) + микро-отклонение
        self.data.qpos[2] = np.pi + np.random.uniform(-0.05, 0.05) 
        
        # 2. ЖЕСТКО ОБНУЛЯЕМ ВСЕ СКОРОСТИ И ИМПУЛЬСЫ (qvel)
        # Это исключит закручивание и бешеные рывки при респауне
        self.data.qvel[0] = 0.0  # Скорость тележки = 0
        self.data.qvel[1] = 0.0  # Угловая скорость колена 1 = 0
        self.data.qvel[2] = 0.0  # Угловая скорость колена 2 = 0
        
        # Синхронизируем измененные координаты и скорости с физическим движком
        mujoco.mj_forward(self.model, self.data)
        
        # Очищаем буфер задержек, так как это новая сессия
        self.action_buffer = [] 
        
        obs = self._get_obs()
        return obs, {}


    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                # Открываем интерактивное 3D-окно MuJoCo
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            # Обновляем сцену текущими физическими данными
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

