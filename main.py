import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import torch


# Define un callback personalizado
class CustomTensorBoardCallback(BaseCallback):
    def __init__(self, log_dir: str, verbose=0):
        super(CustomTensorBoardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_count = 0
        self.episode_reward = 0

    def _on_step(self) -> bool:
        # Accede a la recompensa de este paso
        reward = self.locals.get("rewards", 0)
        self.episode_reward += reward

        # Verifica si el episodio ha terminado
        done = self.locals.get("dones", False)
        if done:
            self.episode_rewards.append(self.episode_reward)
            self.episode_count += 1
            
            # Guarda el promedio de recompensa en TensorBoard
            self.logger.record('episode/reward', self.episode_reward)
            self.logger.record('episode/length', self.episode_count)
            self.logger.dump(self.num_timesteps)

            # Reinicia la recompensa del episodio
            self.episode_reward = 0

        return True

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
policy_kwargs = dict(
        net_arch = [64,64,64],
)

# Crea el entorno
env = gym.make("ALE/Pong-v5")

device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"we are using {device}")
model = PPO(
    "CnnPolicy",  
    env,
    verbose=1,
    tensorboard_log=log_dir,
    policy_kwargs=policy_kwargs,
    learning_rate=0.000001,  
    target_kl=0.01,
    n_steps=2048, 
    batch_size=64,  
    n_epochs=10,
    device=device  #
)

# Define el callback para TensorBoard
callback = CustomTensorBoardCallback(log_dir=log_dir)

# Entrena el modelo y registra los logs
model.learn(total_timesteps=500000, callback=callback)

# Guarda el modelo
model.save("pong_500k_ppo_newparameter")
