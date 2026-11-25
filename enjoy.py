import os
import time
import mujoco.viewer
from stable_baselines3 import PPO
from sim2real import Real2SimPendulumEnv


def main():
    # Создаем среду с явным указанием human-режима
    env = Real2SimPendulumEnv(render_mode="human")
    obs, info = env.reset()

    model_path = None
    for candidate in ("./logs/best_model.zip", "./logs/final_model.zip", "./logs/interrupted_model.zip"):
        if os.path.exists(candidate):
            model_path = candidate
            break
    if model_path is None:
        print("Модель не найдена. Показываем случайные действия.")
        model = None
    else:
        print(f"Загрузка модели из {model_path}...")
        model = PPO.load(model_path)

    print("Запуск пассивного окна MuJoCo...")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            start_time = time.time()

            if model is not None:
                action, _states = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()  # Если ИИ еще не обучен

            obs, reward, terminated, truncated, info = env.step(action)

            if (terminated or truncated) and viewer.is_running():
                obs, info = env.reset()

            # Синхронизируем графику
            viewer.sync()

            # Держим 200 Гц
            elapsed = time.time() - start_time
            if elapsed < 0.005:
                time.sleep(0.005 - elapsed)


if __name__ == "__main__":
    main()
