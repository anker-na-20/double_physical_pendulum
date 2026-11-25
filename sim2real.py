"""Среда-эмулятор физики одноколенного инвертированного маятника на каретке (Sim2Real).

Сервисный цикл ESP32S3 читает датчики (AS5600 → углы) и отправляет PWM на драйвер TMC2209.
Среда учитывает всё: задержки на пути «чтение датчиков → ШИМ», шум датчика,
битность ШИМ (10 бит по умолчанию у ESP32), частоту контроллера (200 Гц)
и физические ограничения мотора.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer


class Real2SimDoublePendulumEnv(gym.Env):
    """
    Custom env for the single inverted pendulum, tuned so that an agent trained here
    transfers 1:1 to the real ESP32S3 + TMC2209 setup.
    (Имя класса оставлено прежним, чтобы не ломать импорты в train.py и enjoy.py)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(self,
                 render_mode=None,
                 xml_path="double_inverted_pendulum.xml",
                 episode_max_steps=1000,
                 init_pole_offset=0.05,
                 max_force=15.0,
                 pwm_bits=10,
                 latency_steps=2,
                 noise_std=0.005,
                 vel_noise_std=0.05):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.xml_path = xml_path

        # --- Sim2Real параметры (настройки железной части) ---
        self.dt = 0.005                      # 200 Гц — частота опроса на ESP32S3
        self.latency_steps = latency_steps   # Задержка на 2 шага (~10 мс) — сенсор→актуатор
        self.noise_std = noise_std           # Шум энкодера AS5600 (~0.3° ≈ 0.005 рад)
        self.vel_noise_std = vel_noise_std   # Шум скоростей (влетает из дифф. позиций)
        self.max_force = max_force           # Ограничение по моменту мотора (Ньютоны)

        # --- Битность ШИМ (hardware fidelity): ESP32 генерирует PWM 10 бит (0..1023) ---
        self.pwm_bits = pwm_bits
        self.pwm_levels = 2 ** self.pwm_bits - 1

        # --- Физическая модель MuJoCo ---
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.dt

        # Пределы рельса берутся из модели (joint range slider)
        slider_id = self.model.joint("slider").id
        self.rail_low, self.rail_high = (float(x) for x in self.model.jnt_range[slider_id])

        # --- Пространство состояний (Observation Space) ---
        # Было 8 параметров, стало 5: [позиция_тележки, sin(θ), cos(θ), скорость_тележки, угловая_скорость]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # --- Состояние среды ---
        self.action_buffer = []
        self.episode_max_steps = episode_max_steps
        self._elapsed_steps = 0
        self.init_pole_offset = init_pole_offset
        self.viewer = None

    # ------------------------------------------------------------------
    def _get_obs(self):
        """Наблюдения из MuJoCo + шум энкодера AS5600."""
        cart_pos  = float(self.data.joint("slider").qpos[0])
        theta     = float(self.data.joint("hinge").qpos[0])
        cart_vel  = float(self.data.joint("slider").qvel[0])
        theta_dot = float(self.data.joint("hinge").qvel[0])

        # Наложение шума энкодера AS5600 (~0.3° ≈ 0.005 рад)
        cart_pos  += float(self.np_random.normal(0, self.noise_std))
        theta     += float(self.np_random.normal(0, self.noise_std))
        cart_vel  += float(self.np_random.normal(0, self.vel_noise_std))
        theta_dot += float(self.np_random.normal(0, self.vel_noise_std))

        return np.array([
            cart_pos,
            np.sin(theta), np.cos(theta),
            cart_vel, theta_dot
        ], dtype=np.float32)

    def step(self, action):
        # 1) Action PWM-квантование (ESP32 PWM 10 бит)
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        pwm = np.round(action * self.pwm_levels) / self.pwm_levels

        # 2) Action Latency: задержка сигнала (чтение датчиков → отправка ШИМ на мотор)
        self.action_buffer.append(pwm)
        if len(self.action_buffer) > self.latency_steps:
            applied = self.action_buffer.pop(0)
        else:
            applied = np.zeros(1, dtype=np.float32)

        # 3) Масштабируем действие под возможности реального мотора + saturation
        u = float(np.clip(applied[0], -1.0, 1.0) * self.max_force)
        self.data.ctrl[0] = u

        # 4) Шаг физического симулятора
        mujoco.mj_step(self.model, self.data)
        self._elapsed_steps += 1

        # 5) Считаем наблюдения (с шумом датчиков)
        # obs = [cart_pos, sin_t, cos_t, cart_vel, t_dot]
        obs = self._get_obs()
        cart_pos = obs[0]
        cos_t    = obs[2]

        # 6) Награда: максимизируем cos(θ) (вертикально вверх: θ = 0 → cos(0) = +1.0)
        r_pose    = 2.0 * cos_t
        r_control = -0.01 * float(np.square(applied)[0])
        r_bounds  = -0.5 * float(np.square(cart_pos))

        reward = float(r_pose + r_control + r_bounds)

        # 7) Отказоустойчивость: тележка уехала за пределы рельса или исчерпан эпизод
        terminated = bool(abs(cart_pos) >= self.rail_high)
        truncated  = bool(self._elapsed_steps >= self.episode_max_steps)
        if terminated:
            reward -= 100.0

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._elapsed_steps = 0

        # Полностью сбрасываем состояние модели и данные
        mujoco.mj_resetData(self.model, self.data)

        # 1. Задаём начальные ПОЗИЦИИ (qpos):
        #    В нашем XML стержень направлен вверх по оси +Z (fromto="0 0 0 0 0 0.4"),
        #    поэтому θ = 0.0 — это положение ВВЕРХ, а θ = π — ВНИЗ (для swing-up).
        offset = options.get("init_pole_offset", self.init_pole_offset) if options else self.init_pole_offset
        self.data.qpos[0] = 0.0  # Тележка строго по центру рельса
        self.data.qpos[1] = np.pi + float(self.np_random.uniform(-offset, offset))

        # 2. ЖЕСТКО ОБНУЛЯЕМ ВСЕ СКОРОСТИ И ИМПУЛЬСЫ (qvel)
        self.data.qvel[0] = 0.0
        self.data.qvel[1] = 0.0
        self.data.ctrl[0] = 0.0

        # Синхронизируем изменённые координаты и скорости с физическим движком
        mujoco.mj_forward(self.model, self.data)

        # Очищаем буфер задержек
        self.action_buffer = []

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
