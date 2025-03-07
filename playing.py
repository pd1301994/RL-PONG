import gym
import pygame
import numpy as np
import time
import cv2
from stable_baselines3 import PPO

# Inicializar pygame
pygame.init()
window = pygame.display.set_mode((300, 200))  # Ventana para detectar teclas

# Crear entorno en modo de renderizado por frames
env = gym.make("ALE/Pong-v5", render_mode='rgb_array')  # Controlamos player 1
model = PPO.load("pong_21M_ppo_newparameters")

# Mapeo de acciones de Pong
ACTIONS = {
    "NOOP": 0,      # No hacer nada
    "FIRE": 1,      # Iniciar la partida
    "UP": 2,        # Subir
    "DOWN": 3       # Bajar
}

def get_human_action():
    """Detecta las teclas presionadas y devuelve la acción correspondiente para el jugador 1."""
    keys = pygame.key.get_pressed()  # Obtener el estado actual de todas las teclas
    if keys[pygame.K_w]:  # Si la tecla 'w' está presionada
        return ACTIONS["UP"]
    elif keys[pygame.K_s]:  # Si la tecla 's' está presionada
        return ACTIONS["DOWN"]
    return ACTIONS["NOOP"]  # Si no se presiona ninguna tecla, no hace nada

num_episodes = 1
for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # 🔹 Acción del humano (jugador 1 - lado izquierdo)
        human_action = get_human_action()

        # 🔹 Enviar solo la acción del jugador 1 (el jugador 2 lo maneja la IA del juego)
        result = env.step(human_action)

        obs, reward, done, truncated, info = result

        total_reward += reward
        
        # Renderizar el entorno en un array
        frame = env.render()

        # Redimensionar la imagen con OpenCV para que se vea más pequeña
        resized_frame = cv2.resize(frame, (600, 600))  
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
        
        # Mostrar la imagen en una ventana
        cv2.imshow("Pong", resized_frame)
        cv2.waitKey(1)

        # 🔹 Pausa para velocidad realista
        time.sleep(0.03)

    print(f"Episodio {episode + 1}: Recompensa total = {total_reward}")

# Cerrar todo
env.close()
cv2.destroyAllWindows()
pygame.quit()
