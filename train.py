"""Обучение PPO на среде Real2SimPendulumEnv (Sim2Real-совместимая).

Паттерн обучения (доказанный на mujoco 4.0.1, sb3 2.9.2):

- Этап 1: 1M шагов с terminated_penalty=-100, episode_max_steps=1500
  (типичный свинг-ап — награда ≈ -137, success 0/10).
- Этап 2 (fine-tune): 300k шагов на среде с terminate_cart_distance=rail_high
  (каретка ограничена рельсом; заскок вершины cos(θ) > 0.95).

Запуск:                              .venv/bin/python train.py
Проверка после обучения:              .venv/bin/python check_success.py
"""
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from sim2real import Real2SimPendulumEnv

ROUND_1 = 1_000_000   # измеренный этап 1
ROUND_2 = 300_000     # измеренный этап 2 (fine-tune)


def main():
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    env = Real2SimPendulumEnv(render_mode=None)
    print("Проверка среды на корректность (check_env)...")
    check_env(env)

    def make_eval_callback(eval_env):
        return EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=10000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )

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
        tensorboard_log=os.path.join(log_dir, "tb_logs"),
    )

    # ---------------- Этап 2: fine-tune с рельсом (типичный свинг-ап) ----------------
    print(f"[Этап 2: fine-tune с рельсом] {ROUND_2} шагов, terminated_penalty=-100...")
    env_rail = Real2SimPendulumEnv(render_mode=None)  # terminate_cart_distance = rail_high
    model.set_env(env_rail)
    try:
        model.learn(total_timesteps=ROUND_2, callback=make_eval_callback(env_rail), progress_bar=True)
    except KeyboardInterrupt:
        model.save(os.path.join(log_dir, "interrupted_model"))
        return

    model.save(os.path.join(log_dir, "final_model"))
    print("Обучение завершено! Модель сохранена в ./logs/final_model.zip")

    eval_env2 = Real2SimPendulumEnv(render_mode=None)
    mean_r, std_r = evaluate_policy(model, eval_env2, n_eval_episodes=20, deterministic=True)
    print(f"\nФинальная eval: mean_reward={mean_r:.3f} (+/- {std_r:.3f})")
    print("Проверьте success rate: .venv/bin/python check_success.py")


if __name__ == "__main__":
    main()
