"""Среда-эмулятор физики одноколенного инвертированного маятника на каретке (Sim2Real).

Сервисный цикл ESP32S3 читает датчики (AS5600 → угол) и отправляет PWM на драйвер TMC2209.
Среда учитывает всё: задержки на пути «чтение датчиков → ШИМ», шум датчика,
битность ШИМ (10 бит по умолчанию у ESP32), частоту контроллера (200 Гц)
и физические ограничения мотора.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer


class Real2SimPendulumEnv(gym.Env):
    """
    Custom env for the single-knee inverted pendulum, tuned so that an agent trained here
    transfers 1:1 to the real ESP32S3 + TMC2209 setup.

    Наблюдения (5): [позиция_тележки, sin(θ), cos(θ), скорость_тележки, θ_dot].
    Цель (swing-up): поднять колено наверх (cos(θ) → +1), как и прописано в reward.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(self,
                 render_mode=None,
                 xml_path="double_inverted_pendulum.xml",
                 episode_max_steps=3000,
                 init_pole_offset=0.05,
                 max_force=15.0,
                 pwm_bits=10,
                 latency_steps=2,
                 noise_std=0.005,
                 vel_noise_std=0.05,
                 terminate_cart_distance=None,
                 terminated_penalty=-100.0):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.xml_path = xml_path

        # --- Sim2Real параметры (настройки железной части) ---
        self.dt = 0.005                # 200 Гц — частота опроса на ESP32S3
        self.latency_steps = latency_steps   # Задержка на 2 шага (~10 мс) — сенсор→актуатор
        self.noise_std = noise_std     # Шум энкодера AS5600 (~0.3° ≈ 0.005 рад)
        self.vel_noise_std = vel_noise_std  # Шум скоростей (влетает из дифф. позиций)
        self.max_force = max_force     # Ограничение по моменту мотора (Ньютоны)

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
        self.terminate_cart_distance = float(terminate_cart_distance) if terminate_cart_distance else self.rail_high
        self.terminated_penalty = terminated_penalty

        # --- Пространство состояний (Observation Space) ---
        # [позиция_тележки, sin(θ), cos(θ), скорость_тележки, θ_dot] — 5 переменных
        # sin/cos — исключают разрыв при переходе через 360°.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # --- Состояние среды ---
        self.action_buffer = []          # Буфер для эмуляции задержки сигнала
        self.episode_max_steps = episode_max_steps  # Отказоустойчивость: трансформ в truncated
        self._elapsed_steps = 0
        self.init_pole_offset = init_pole_offset
        self.viewer = None

    # ------------------------------------------------------------------
    def _get_obs(self):
        """Наблюдения из MuJoCo + шум энкодера AS5600 (на скорости — дифференциальный шум)."""
        cart_pos   = float(self.data.joint("slider").qpos[0])
        theta      = float(self.data.joint("hinge1").qpos[0])
        cart_vel   = float(self.data.joint("slider").qvel[0])
        theta_dot  = float(self.data.joint("hinge1").qvel[0])

        # Наложение шума энкодера AS5600 (~0.3° ≈ 0.005 рад)
        cart_pos   += float(self.np_random.normal(0, self.noise_std))
        theta      += float(self.np_random.normal(0, self.noise_std))
        cart_vel   += float(self.np_random.normal(0, self.vel_noise_std))
        theta_dot  += float(self.np_random.normal(0, self.vel_noise_std))

        return np.array([
            cart_pos,
            np.sin(theta), np.cos(theta),
            cart_vel, theta_dot
        ], dtype=np.float32)

    def step(self, action):
        # 1) Action PWM-квантование (befits real hardware: ESP32 PWM 10 бит)
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        pwm = np.round(action * self.pwm_levels) / self.pwm_levels   # 10-битный ШИМ

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
        obs = self._get_obs()

        # СТРОГО фиксируем индексы из вектора obs:
        # obs = [cart_pos, sin_theta, cos_theta, cart_vel, theta_dot]
        cart_pos = obs[0]
        cos_theta = obs[2]

        # 6) Награда — паттерн swing-up: колено наверху (cos(θ) → +1)
        r_pose    = 1.0 * cos_theta                    # + когда колено "вверх"
        # Бонус за прохождение вертикали ("с рывком"): постепенно наращиваем награду
        # по мере подъёма колена — помогает агенту найти резонанс свинга и совершить рывок наверх.
        r_swing   = 0.6 if cos_theta > 0.8 else (0.3 if cos_theta > 0.5 else 0.0)
        r_control = -0.01 * float(np.square(applied)[0])  # дешевка на применённое действие
        r_bounds  = -0.5 * float(np.square(cart_pos))

        reward = float(r_pose + r_swing + r_control + r_bounds)

        # 7) Отказоустойчивость: тележка уехала за пределы рельса или исчерпан эпизод
        terminated = bool(abs(cart_pos) >= self.terminate_cart_distance)
        truncated  = bool(self._elapsed_steps >= self.episode_max_steps)
        if terminated:
            reward += self.terminated_penalty

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._elapsed_steps = 0

        # Полностью сбрасываем состояние модели и данные
        mujoco.mj_resetData(self.model, self.data)

        # 1. Задаём начальные ПОЗИЦИИ (qpos): тележка по центру, θ = π → колено вниз ("спавн с коленом внизу")
        offset = options.get("init_pole_offset", self.init_pole_offset) if options else self.init_pole_offset
        self.data.qpos[0] = 0.0                                  # Тележка строго по центру рельса
        self.data.qpos[1] = np.pi + float(self.np_random.uniform(-offset, offset))

        # 2. ЖЕСТКО ОБНУЛЯЕМ ВСЕ СКОРОСТИ И ИМПУЛЬСЫ (qvel)
        self.data.qvel[:] = 0.0
        self.data.ctrl[0] = 0.0

        # Синхронизируем изменённые координаты и скорости с физическим движком
        mujoco.mj_forward(self.model, self.data)

        # Очищаем буфер задержек, так как это новая сессия
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
