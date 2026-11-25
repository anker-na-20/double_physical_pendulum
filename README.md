# Двойной инвертированный маятник на каретке

Реализация системы: рельс (50 см), каретка (NEMA-17, трос, TMC2209), два колена (по 20 см),
магнитный энкодер AS5600 → ESP32S3, на нём обученная сеть (PPO) выдаёт скорость мотора
и удерживает вершину колен наверху.

> **Важно:** модель тренируется "с нуля" в положении «вершины колен вниз» (swing-up) — это
> классический запуск двойного инвертированного маятника (DIP). Подробности: `ANALYSIS.md`.

## Файлы

- `double_inverted_pendulum.xml` — физика MuJoCo (длина колена 20 см + грузик, рельс ±0.25 м)
- `sim2real.py` — Gym-env с учётом Sim2Real: задержки (200 Гц, ~10 мс), шум энкодера, PWM 10 бит, лимиты мотора
- `train.py` — обучение PPO, TensorBoard, EvalCallback, `check_env`
- `enjoy.py` / `run_visual.py` — демо-циклы с визуализацией
- `export_weights.py` — экспорт весов PPO → `ppo_policy.h/.c` для ESP32S3

## Установка и обучение

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # см. pyproject.toml (mujoco, gymnasium, sb3, tensorboard, tqdm, rich)
python train.py
```

- Обучение: 300 000 шагов (см. замеры этапов в `ANALYSIS.md` — свинг-ап с PPO
  достигает 0/10, успех не получен).
- Лучшая модель (по eval) → `logs/best_model.zip`, последняя → `logs/final_model.zip`.
- TensorBoard: `tensorboard --logdir logs/tb_logs` — ключевая метрика: `mean_reward` растёт
  и верх кривой (`cos(θ1)≈cos(θ2)≈+1`) означает: вершина колен наверху.

## Деплой на ESP32S3

```bash
python export_weights.py logs/best_model.zip   # -> ppo_policy.h + ppo_policy.c (3 слоя, ~90 КБ)
```

Подключите `ppo_policy.h/.c` в проект ESP:

```c
#include "ppo_policy.h"
float obs[PPO_OBS_DIM] = {cart_pos, sin_t1, cos_t1, sin_t2, cos_t2, cart_vel, t1_dot, t2_dot};
float action[PPO_ACT_DIM];
ppo_policy_forward(obs, action);        // action ∈ [-1, 1]
int pwm = (int)round(action[0] * 1023); // 10-битный PWM, как в sim2real.py
```

Цикл раз в ~5 мс (200 Гц) — совпадает с `sim2real.py` (`dt=0.005`), задержка на пути
«чтение датчиков → ШИМ» ≈ 2 шага (10 мс). Оцените реальный максимум момента NEMA-17
и внесите его в `max_force` (`sim2real.py`) и `ctrlrange` (`double_inverted_pendulum.xml`).

## Запуск демо

```bash
python enjoy.py       # лучшая модель из logs/ в интерактивном MuJoCo-окне
python run_visual.py   # случайные действия, отладка физики
```
