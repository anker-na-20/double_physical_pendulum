#!/usr/bin/env python3
"""Оценить success rate обученной модели: удерживается ли колено наверху (cos(θ) > 0.95).

Запуск: .venv/bin/python check_success.py [путь_к_модели]   (по умолчанию logs/best_model.zip)
"""
import sys

import numpy as np
from stable_baselines3 import PPO

from sim2real import Real2SimPendulumEnv


def success_rate(model_path="logs/best_model.zip", n=10, total_steps=2000):
    model = PPO.load(model_path)
    env = Real2SimPendulumEnv(render_mode=None)
    ok = 0
    for t in range(n):
        obs, _ = env.reset()
        for i in range(total_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            if float(obs[2]) > 0.95:  # cos(θ) > 0.95 — колено наверху
                ok += 1
                break
            if term or trunc:
                break
    return ok, n


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/best_model.zip"
    ok, n = success_rate(path)
    print(f"success rate ({path}): {ok}/{n} — колено наверху cos(θ)>0.95")
