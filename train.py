import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from sim2real import Real2SimDoublePendulumEnv

def main():
    # 1. Создаем изолированные среды для обучения и тестов (без графики для скорости)
    env = Real2SimDoublePendulumEnv(render_mode=None)
    eval_env = Real2SimDoublePendulumEnv(render_mode=None)

    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # 2. Настраиваем Callback. Каждые 10 000 шагов ИИ проверяется на чистом треке
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=log_dir,
        log_path=log_dir, 
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    print("Инициализация PPO с поддержкой TensorBoard...")
    # 3. Конфигурируем нейросеть
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        batch_size=64,
        n_steps=2048,           
        tensorboard_log=os.path.join(log_dir, "tb_logs") # Сюда пишутся живые графики
    )

    print("Старт обучения на 1 000 000 шагов. Для прерывания нажмите Ctrl+C.")
    try:
        model.learn(total_timesteps=1000000, callback=eval_callback)
        print("Обучение завершено! Модель сохранена в ./logs/best_model.zip")
    except KeyboardInterrupt:
        print("\nПроцесс прерван. Сохраняем текущую промежуточную модель...")
        model.save(os.path.join(log_dir, "interrupted_model"))
        
if __name__ == "__main__":
    main()
