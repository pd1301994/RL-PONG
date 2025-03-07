# RL-Pong

**Important:** Python = 3.9 is required!

## Steps
1. Run the following command to install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Use this link to download the trained model: https://drive.google.com/file/d/1625ZC9GQFoK-E8D_sLIXkPAb3TN3oQ4G/view?usp=sharing

   ```
   logs
   eval.py
   main.py
   playing.py
   retrain.py
   pong_21M_ppo_newparameters
   ```
### Make sure everything is in the same folder
## File Descriptions
- **eval.py**: This file will conduct a competition against the Gymnasium agent. The user should not touch anything.
- **main.py**: This is the program where the first trained model is defined.
- **playing.py**: The user will compete against the model. Use the **w** and **s** keys (up and down).
- **retrain.py**: This file is used to retrain an existing model. It has a checkpoint in case the user wants to cancel the training mid-execution. A model is loaded first and then retrained.
- **pong_21M_ppo_newparameters**: This is the model already trained in PPO. It is stored on Drive because it takes up a lot of space and cannot be uploaded to Git.
- **logs**: This folder contains several subfolders from different logs. Use this command if you want to visualize the project:
  ```bash
  tensorboard --logdir=logs
  ```

## Questions and Suggestions
For any questions or suggestions, please contact: [ps.dedios94@gmail.com](mailto:ps.dedios94@gmail.com)
