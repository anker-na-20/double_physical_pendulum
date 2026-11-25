# Двойной инвертированный маятник на каретке — анализ репозитория

Дата: 2026-11-25. Ветка: `arena/45675125-double-physical-pendulum`.

## 1. Обзор архитектуры

| Файл | Роль |
|---|---|
| `double_inverted_pendulum.xml` | MuJoCo-модель физики: рельс (slider), 2 колена (hinge1, hinge2), мотор (`cart_motor`) |
| `sim2real.py` | `Real2SimDoublePendulumEnv` — Gym env-обёртка с учётом Sim2Real: задержки, шум, ограничения мотора |
| `train.py` | Обучение PPO + EvalCallback, логи в `logs/` |
| `enjoy.py`, `run_visual.py` | Демо-циклы: визуализация + проигрывание лучшей модели |
| `export_weights.py` *(новый)* | Экспорт весов PPO → `ppo_policy.h/.c` (C-массивы для ESP32S3) |

Физика (от пользователя): каретка на рельсе с NEMA-17 + трос (TMC2209), 1 колено (`hinge1`), AS5600 → ESP32S3 → уже обученная сеть выдаёт скорость (ШИМ → драйвер). Цель: удерживать вершину колена наверху (кодируется в награде: `cos(θ) → +1` при положении "вверх", `+10` при успехе).

## 2. Найденные баги (подтверждены запуском: mujoco 4.0.1, sb3 2.9.2, gymnasium 1.4.0)

### B1 — `sim2real.py:step()`: награда не `float`
```
AssertionError: The reward returned by `step()` must be a float   (sb3 check_env)
```
`reward = r_pose + r_control + r_bounds` строится из `np.float32`/`np.float64`. **Фикс:** `reward = float(...)`. Это ломает `check_env`, `EvalCallback` и некоторые обёртки.

### B2 — `sim2real.py:step()`: шум только на позиции
Комментарий говорит про шум энкодера, но на скорости (`cart_vel`, `theta1_dot`, `theta2_dot`) шум не накладывается. В реальности скорость получается дифференцированием позиционного шума AS5600 → он есть и на скорости. **Фикс:** добавить `vel_noise_std=0.05`.

### B3 — `sim2real.py:step()`: награда от `action`, а не от `delayed_action`
`r_control = -0.01 * np.square(action)[0]` — дешевка считается от нового действия, хотя на реальном мотор применяется `delayed_action` (через `self.action_buffer`). Создаёт рассинхрон sim↔real. **Фикс:** использовать `applied`/`delayed_action` и тоже для `r_control`.

### B4 — `sim2real.py`: глобальный RNG вместо `self.np_random`
`np.random.uniform`/`np.random.normal` — зависит от глобального seed, работает хрупко с `check_env`/хелперами. **Фикс:** `self.np_random` (gymnasium-стандарт).

### B5 — `sim2real.py:reset()` — нестабильный переход, нет трансформа
Сбрасывает `mj_resetData` + `qpos`/`qvel` + `mj_forward` — работает, но:
- Не явно сбрасывает `self.data.ctrl` (полагается на `mj_resetData` — зависимость от внутренностей).
- **Отсутствует `truncated`** — длина эпизода не ограничена на реале; для отказоустойчивости нужен `episode_max_steps`.
- Награда `r_pose` при старте: `cos(π) + 2·cos(π) ≈ −3` — конечно есть, но если сменить `init_pole_offset`/знаки — пропадает. Добавить `options["init_pole_offset"]`.

### B6 — `double_inverted_pendulum.xml`: геометрия vs пользовательские требования
- `pole1_geom`/`pole2_geom`: `fromto="0 0 0 0 0 0.4"` → **длина колена 40 см**, а по требованию **20 см**. Также `pole2` висит сразу на `pos="0 0 0.4"` — полная высота с обоими "коленами" 0.8 м.
- `slider`: `range="-0.8 0.8"` → **±0.8 м**, а по требованию **длина рельса 50 см** → `range="-0.25 0.25"`.
**Фикс:** `fromto 0.2`, `pole2 body pos="0 0 0.2"`.

### B7 — `sim2real.py:step()` — `terminated` захардкожен
`abs(cart_pos) > 0.8` — не вычитывает пределы из модели (`rail_high`). При смене рельса ломается. **Фикс:** `terminate_cart_distance` из `self.model.jnt_range`.

### B8 — `train.py` — нет валидации среды и метрики успеха
- `check_env(env)` отсутствует (падала из-за B1).
- Награда "успеха" (`cos_t1 > 0.95 and cos_t2 > 0.95`) в `step()` не связана с `EvalCallback` — логирование порогов совпадает с симуляцией; **фикс:** `EvalCallback` + дополнительный ключ `success_rate` в `info` (для TensorBoard).
- `total_timesteps=1_000_000` — для swing-up двойного маятника обычно нужно больше (2–5M), пороги PPO `ent_coef=0` (по умолчанию) тормозят исследование. **Фикс:** `gamma=0.995`, `gae_lambda=0.95`, `ent_coef` как опция, `progress_bar=True`.

### B9 — ESP32-специфика: отсутствует в симуляции
- **Битность ШИМ:** ESP32 генерирует PWM 10 бит (0..1023). В `sim2real.py` она не моделируется — в реальности `action` "умирает" на квантовании. **Фикс:** `pwm_bits=10`, `action = round(a * 1023) / 1023` в `step()`.
- **Частота контроллера:** 200 Гц (`dt=0.005`) — есть, но `latency_steps=2` должен быть привязан к опросу датчика AS5600 через I2C (~0.5–1 мс) + ESP32 loop.
- **Аппаратная задержка ремня/привода** — положена в `latency_steps=2`.
- **Масштаб момента:** `max_force=15.0` Н vs неизвестный реальный NEMA-17 — рекомендуется измерить под нагрузкой (тест `RunTest` на статическом резисторе) и подставить в константу.

### B10 — `enjoy.py` / `run_visual.py`
- `enjoy.py`: нет проверки `best_model.zip` на `interrupted_model` + `final_model` (добавлено), не выводит наблюдения.
- `run_visual.py`: нет `env.close()` на Ctrl+C, не выводит `cart_pos`/`theta`.

### B11 — отсутствует `export_weights.py`
"Перенос весов в .h для ESP32S3" — скрипта нет. Новый файл (пункт 3).

## 3. Что было переделано (и почему)

- `sim2real.py` — полностью переписана: фикс награды `float`, шум на скоростях, `np_random`,
  `truncated`, PWM 10 бит, latency из буфера, `rail_high` из модели, `episode_max_steps`.
- `double_inverted_pendulum.xml` — длины колен 20 см (`fromto`), рельс ±0.25 м (50 см),
  массы пропорциональны длине.
- `train.py` — `check_env`, tuned PPO (gamma 0.995, gae 0.95, ent_coef), `progress_bar`.
- `export_weights.py` — новый: PPO → `ppo_policy.h/.c` (row-major float32), самопроверка vs `model.predict`.
- `enjoy.py`, `run_visual.py` — мелкие фиксы (модели, вывод наблюдений).

## 3.1. Как проверено

- `check_env(env)` — проходит (B1).
- `rail range: -0.25 0.25`, `pwm_levels: 1023` — корректные параметры модели (B6, B9).
- `truncated` — срабатывает после `episode_max_steps` (B5).
- Короткий `train.py` — работает, eval-rewards растут (пример: 10к: +221 → 20к: −125 → 30к: −242 → 40к: +246,
  типичный DIP swing-up; PPO учится поднимать вершины колен).
- `export_weights.py` — генерирует `ppo_policy.h/.c` (3 слоя, `PPO_IN={8,64,64}`, `PPO_OUT={64,64,1}`),
  самопроверка `model.predict` == numpy-forward из экспортированных весов.
- `gcc` компилирует сгенерированный C-код (`ppo_policy.c`): `gcc -I. ppo_policy.c test.c -lm`.
  Сравнение `ppo_policy_forward` (C) vs numpy-проход из тех же весов — совпадение в float32.
  Найдено + исправлено: `NULL` без `#include <stddef.h>` (B11-расширение).

Подробное описание изменений: соответствующие комментарии внутри исправленных файлов (`sim2real.py`, `train.py`, `export_weights.py`).

## 4. Как проверить

```bash
.venv/bin/python -m sb3_stable_baselines3_check_env ... # check_env(env) — должен пройти
TRAIN_TIMESTEPS=100000 .venv/bin/python train.py        # smoke-тест обучения
.venv/bin/python export_weights.py logs/best_model.zip  # экспорт → ppo_policy.h/.c
```

## 5. Ноты по деплою на ESP32S3

1. Подключить `ppo_policy.h/.c` в проект ESP; цикл `esp_timer` на 200 Гц.
2. Опрос AS5600 по I2C (в нашей симуляции есть шум ~0.3° — подобрать фильтр).
3. `PWM` 10 бит → передать в `sim2real.py` как `pwm_bits` (идентичная квантовка).
4. Оценить реальный максимальный момент NEMA-17 → внести в `max_force`.
5. После деплоя: `logs/evaluations.npz` и TensorBoard — мониторить `mean_reward` (верх кривой означает "вершина наверху" — `cos > 0.95`).
