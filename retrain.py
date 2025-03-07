import sys
import gym
from stable_baselines3 import PPO
import os
import signal
import sys
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


# Configura el directorio de logs
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

# Crea el entorno
env = gym.make("ALE/Pong-v5")

# Verifica si CUDA está disponible y asigna el dispositivo
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"The device is{device}")
# Cargar el modelo previamente entrenado
model = PPO.load("pong_16M_ppo_checkpoint", env=env, device=device)
model_save_path = "pong_16M_ppo_checkpoint"


# Función para manejar la señal de interrupción y guardar el modelo
def save_model_on_interrupt(signal, frame):
    print("\nInterrupción recibida. Guardando el modelo...")
    model.save(model_save_path)
    print(f"Modelo guardado en {model_save_path}")
    sys.exit(0)  # Finaliza el programa

# Asociar la señal SIGINT (Ctrl+C) a la función de guardado
signal.signal(signal.SIGINT, save_model_on_interrupt)
# Ahora continúas entrenando con más pasos

 
for param_group in model.policy.optimizer.param_groups:
    param_group['lr'] = 1e-7 # Ajusta la tasa de aprendizaje aquí
model.target_kl=0.01

# Verifica que se haya cambiado la tasa de aprendizaje
print(f"Learning rate actual: {model.policy.optimizer.param_groups[0]['lr']}")

model.learn(total_timesteps=5000000, callback=None)  # Aquí pones 500k pasos adicionales

# Guarda el modelo después de continuar el entrenamiento
model.save("pong_21M_ppo_newparameters")  # Este es el nuevo modelo después de 1 millón de pasos
