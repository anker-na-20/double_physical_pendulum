#!/usr/bin/env python3
"""Экспорт весов обученной политики PPO в C-массивы для ESP32S3 (motor loop).

Использование:
    .venv/bin/python export_weights.py [путь_к_модели]          # logs/best_model.zip по умолчанию

В рабочей директории появляются:
    ppo_policy.h  — определения, веса (static const float PPO_W0..N), размеры слоёв
    ppo_policy.c  — ppo_policy_forward(obs, action) — прямой проход сети (Tanh между слоями)

На ESP32S3 (Arduino IDE / ESP-IDF):
    #include "ppo_policy.h"
    float obs[PPO_OBS_DIM] = {...};  // [cart_pos, sin_t1, cos_t1, sin_t2, cos_t2, cart_vel, t1_dot, t2_dot]
    float action[PPO_ACT_DIM];
    ppo_policy_forward(obs, action); // action ∈ [-1, 1] → duty-cycle ШИМ (квантование 10 бит)
"""
import sys

import numpy as np
import torch
from stable_baselines3 import PPO


def _collect_layers(model):
    """Список (W, b, tanh_after) — Linear-слои политики (policy_net + action_net)."""
    policy = model.policy
    modules = list(policy.mlp_extractor.policy_net)
    modules.append(policy.action_net)

    layers = []
    for i, m in enumerate(modules):
        if not isinstance(m, torch.nn.Linear):
            continue
        nxt = modules[i + 1] if i + 1 < len(modules) else None
        tanh_after = isinstance(nxt, torch.nn.Tanh)
        W = np.asarray(m.weight.detach().cpu().numpy(), dtype=np.float32)
        b = np.asarray(m.bias.detach().cpu().numpy(), dtype=np.float32)
        layers.append((W, b, tanh_after))
    return layers


def _arr(a):
    return ", ".join(repr(float(v)) for v in a.flatten())


def write_header(layers, obs_dim, act_dim, path="ppo_policy.h"):
    max_hidden = max(w.shape[0] for w, _, _ in layers)
    lines = [
        "#pragma once",
        "#include <stdint.h>",
        f"#define PPO_OBS_DIM {obs_dim}",
        f"#define PPO_ACT_DIM {act_dim}",
        f"#define PPO_LAYER_COUNT {len(layers)}",
        f"#define PPO_MAX_HIDDEN {max_hidden}",
        "",
    ]
    for k, (w, b, _) in enumerate(layers):
        lines.append(f"static const float PPO_W{k}[{w.size}] = {{{_arr(w)}}};")
        lines.append(f"static const float PPO_B{k}[{b.size}] = {{{_arr(b)}}};")
    lines.append(f"static const int PPO_IN[PPO_LAYER_COUNT] = {{{', '.join(str(w.shape[1]) for w, _, _ in layers)}}};")
    lines.append(f"static const int PPO_OUT[PPO_LAYER_COUNT] = {{{', '.join(str(w.shape[0]) for w, _, _ in layers)}}};")
    lines.append(f"static const uint8_t PPO_TANH_AFTER[PPO_LAYER_COUNT] = {{{', '.join('1' if t else '0' for _, _, t in layers)}}};")
    lines.append("")
    lines.append("void ppo_policy_forward(const float *obs, float *action);")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_c(layers, obs_dim, act_dim, path="ppo_policy.c"):
    lines = [
        '#include "ppo_policy.h"',
        "#include <math.h>",
        "#include <stddef.h>",   # NULL для switch/case (и будущих портов)
        "",
        "void ppo_policy_forward(const float *obs, float *action)",
        "{",
        "    static float cur[PPO_MAX_HIDDEN];",
        "    const float *in = obs;",
        "    for (int k = 0; k < PPO_LAYER_COUNT; k++) {",
        "        const float *W = NULL;",
        "        const float *B = NULL;",
        "        switch (k) {",
    ]
    for k in range(len(layers)):
        lines.append(f"        case {k}: W = PPO_W{k}; B = PPO_B{k}; break;")
    lines.append("        }")
    lines.append("        const int inn = PPO_IN[k];")
    lines.append("        const int out = PPO_OUT[k];")
    lines.append("        for (int i = 0; i < out; i++) {")
    lines.append("            float s = B[i];")
    lines.append("            for (int j = 0; j < inn; j++) s += W[i * inn + j] * in[j];")
    lines.append("            cur[i] = s;")
    lines.append("        }")
    lines.append("        if (PPO_TANH_AFTER[k])")
    lines.append("            for (int i = 0; i < out; i++) cur[i] = tanhf(cur[i]);")
    lines.append("        if (k == PPO_LAYER_COUNT - 1) {")
    lines.append(f"            for (int i = 0; i < PPO_ACT_DIM; i++) action[i] = cur[i];")
    lines.append("            return;")
    lines.append("        }")
    lines.append("        in = cur;")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "logs/best_model.zip"
    model = PPO.load(model_path)
    layers = _collect_layers(model)
    obs_dim = int(model.observation_space.shape[0])
    act_dim = int(model.action_space.shape[0])

    write_header(layers, obs_dim, act_dim, "ppo_policy.h")
    write_c(layers, obs_dim, act_dim, "ppo_policy.c")
    print(f"Экспорт модели {model_path} ({len(layers)} слоёв) -> ppo_policy.h / ppo_policy.c")

    # Быстрая самопроверка: numpy-переимплементация C-кода vs model.predict
    def c_forward(obs):
        h = obs.astype(np.float32).reshape(-1)
        for k, (w, b, t) in enumerate(layers):
            h = w @ h + b
            if t:
                h = np.tanh(h)
            if k == len(layers) - 1:
                return h
        return h

    env_obs = None
    try:
        from sim2real import Real2SimDoublePendulumEnv
        env = Real2SimDoublePendulumEnv(render_mode=None)
        obs, _ = env.reset()
        env_obs = obs
        pred_cpp, _ = model.predict(obs, deterministic=True)
        print("self-check:", np.round(c_forward(obs), 4), "vs model:", np.round(pred_cpp.reshape(-1), 4))
    except Exception as e:
        print("self-check skipped:", e)


if __name__ == "__main__":
    main()
