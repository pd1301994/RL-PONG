import time
import gym
import cv2
from stable_baselines3 import PPO

env = gym.make("ALE/Pong-v5", render_mode='rgb_array')
model = PPO.load("pong_21M_ppo_newparameters") 

num_episodes = 3  # Número de episodios a ejecutar
for episode in range(num_episodes):
    obs, info = env.reset()  # Desempaquetamos el estado y la info
    done = False
    total_reward = 0

    while not done:
        # El agente toma una acción aleatoria
        action, _states = model.predict(obs) 
        print(f"Acción tomada: {action}") # Toma una acción aleatoria del espacio de acciones

        # Ejecuta la acción en el entorno
        result = env.step(action)

        # Dependiendo de la versión del entorno, puede devolver 4 o 5 valores
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result  # Para entornos con 4 valores

        # Actualiza la recompensa total
        total_reward += reward
        
        # Renderiza el entorno (visualización)
        frame = env.render()
        resized_frame = cv2.resize(frame, (1024, 1024))
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
        time.sleep(0.01)

        cv2.imshow("Pong", resized_frame)
        cv2.waitKey(1)


    print(f"Episodio {episode + 1}: Recompensa total = {total_reward}")

# Cierra el entorno después de que se terminan los episodios
env.close()
