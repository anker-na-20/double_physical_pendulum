import time
from sim2real import Real2SimPendulumEnv


def main():
    # Инициализируем среду с включенной визуализацией
    env = Real2SimPendulumEnv(render_mode="human")

    # Сбрасываем систему в начальное состояние
    obs, info = env.reset()

    print("Визуализация запущена. Нажмите Ctrl+C в консоли для выхода.")

    try:
        while True:
            start_time = time.time()

            # Генерируем случайное действие (управление мотором от -1.0 до 1.0)
            action = env.action_space.sample()

            # Делаем шаг в симуляторе
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"cart_pos={obs[0]:+.3f} theta1={obs[2]:+.3f} theta2={obs[4]:+.3f} r={reward:+.3f}")

            # Отрисовываем кадр
            env.render()

            # Если тележка врезалась в край рельса — сбрасываем маятник
            if terminated or truncated:
                obs, info = env.reset()

            # Синхронизируем шаг симуляции с реальным временем (200 Гц -> 0.005 сек)
            elapsed = time.time() - start_time
            if elapsed < 0.005:
                time.sleep(0.005 - elapsed)

    except KeyboardInterrupt:
        print("\nОстановка визуализации...")
    finally:
        env.close()


if __name__ == "__main__":
    main()
