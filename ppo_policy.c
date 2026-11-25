#include "ppo_policy.h"
#include <math.h>
#include <stddef.h>

void ppo_policy_forward(const float *obs, float *action)
{
    static float cur[PPO_MAX_HIDDEN];
    const float *in = obs;
    for (int k = 0; k < PPO_LAYER_COUNT; k++) {
        const float *W = NULL;
        const float *B = NULL;
        switch (k) {
        case 0: W = PPO_W0; B = PPO_B0; break;
        case 1: W = PPO_W1; B = PPO_B1; break;
        case 2: W = PPO_W2; B = PPO_B2; break;
        }
        const int inn = PPO_IN[k];
        const int out = PPO_OUT[k];
        for (int i = 0; i < out; i++) {
            float s = B[i];
            for (int j = 0; j < inn; j++) s += W[i * inn + j] * in[j];
            cur[i] = s;
        }
        if (PPO_TANH_AFTER[k])
            for (int i = 0; i < out; i++) cur[i] = tanhf(cur[i]);
        if (k == PPO_LAYER_COUNT - 1) {
            for (int i = 0; i < PPO_ACT_DIM; i++) action[i] = cur[i];
            return;
        }
        in = cur;
    }
}
