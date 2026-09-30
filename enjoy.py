import os
import time
import mujoco.viewer
from stable_baselines3 import PPO
from sim2real import Real2SimDoublePendulumEnv

def main():
    # Создаем среду с явным указанием human-режима
    env = Real2SimDoublePendulumEnv(render_mode="human")
    obs, info = env.reset()
    
    model_path = "./logs/best_model.zip"
    if not os.path.exists(model_path):
        # Если лучшей модели еще нет, возьмем промежуточную (интервальную)
        model_path = "./logs/interrupted_model.zip"
        
    if not os.path.exists(model_path):
        print("Модель не найдена. Показываем случайные действия.")
        model = None
    else:
        print(f"Загрузка модели из {model_path}...")
        model = PPO.load(model_path)
    
    print("Запуск пассивного окна MuJoCo...")
    # Запуск окна в пассивном режиме (идеально для Mac при запуске через mjpython)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            start_time = time.time()
            
            if model is not None:
                action, _states = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample() # Если ИИ еще не обучен
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()
                
            # Синхронизируем графику
            viewer.sync()
            
            # Держим 200 Гц
            elapsed = time.time() - start_time
            if elapsed < 0.005:
                time.sleep(0.005 - elapsed)

if __name__ == "__main__":
    main()
