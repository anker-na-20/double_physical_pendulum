"""Обучение PPO на среде Real2SimDoublePendulumEnv (Sim2Real-совместимая).

Запуск:                              .venv/bin/python train.py
С ограничением шагов (для smoke-теста): TRAIN_TIMESTEPS=100000 .venv/bin/python train.py
"""
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from sim2real import Real2SimDoublePendulumEnv


def main():
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    # 1. Создаем изолированные среды для обучения и тестов (без графики для скорости)
    env = Real2SimDoublePendulumEnv(render_mode=None)
    eval_env = Real2SimDoublePendulumEnv(render_mode=None)

    print("Проверка среды на корректность (check_env)...")
    check_env(env)

    # 2. Настраиваем Callback. Каждые 10 000 шагов ИИ проверяется на чистом треке
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    print("Инициализация PPO с поддержкой TensorBoard...")
    # 3. Конфигурируем нейросеть.
    # Настройки для swing-up двойного инвертированного маятника:
    #  - gamma=0.995: длинные горизонты работы с отложенной наградой за удержание вершины;
    #  - ent_coef=0.01: немного энтропии — исследование при swing-up (без него PPO сходится быстрее, но в тупик);
    #  - learning_rate=3e-4 с декаимацией при помощи сложности задачи — держим фиксированно;
    #  - n_steps=2048 и batch_size=64: стабильные батчи для MlpPolicy.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=os.path.join(log_dir, "tb_logs"),  # Сюда пишутся живые графики
    )

    total_timesteps = int(os.environ.get("TRAIN_TIMESTEPS", "2000000"))
    print(f"Старт обучения на {total_timesteps} шагов. Для прерывания нажмите Ctrl+C.")
    try:
        import rich  # noqa: F401  — progress_bar требует rich
        progress = True
    except ImportError:
        progress = False
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=progress)
        model.save(os.path.join(log_dir, "final_model"))
        print("Обучение завершено! Модель сохранена в ./logs/final_model.zip")
    except KeyboardInterrupt:
        print("\nПроцесс прерван. Сохраняем текущую промежуточную модель...")
        model.save(os.path.join(log_dir, "interrupted_model"))


if __name__ == "__main__":
    main()
