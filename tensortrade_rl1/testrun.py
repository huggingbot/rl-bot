import numpy as np
import ray
from ray.rllib.agents.ppo import ppo
from ray.tune import register_env

from tensortrade_rl1.main import create_env
from tensortrade_rl1.train_config import ENV_NAME, RAY_CONFIG, CHECKPOINT_PATH, MODE


def main():
    assert MODE == 'test' or MODE == 'debug'
    ray.init()
    register_env(ENV_NAME, create_env)

    # Restore agent
    agent = None
    if MODE == 'test':
        agent = ppo.PPOTrainer(env=ENV_NAME, config=RAY_CONFIG)
        agent.restore(CHECKPOINT_PATH)

    env = create_env(RAY_CONFIG['env_config'])

    # Run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()

    # Initialize hidden_state variable that will correspond to lstm_cell_size
    lstm_cell_size = RAY_CONFIG['model']['lstm_cell_size']
    hidden_state = [np.zeros(lstm_cell_size), np.zeros(lstm_cell_size)]

    while not done:
        # In order for use_lstm to work we set full_fetch to True
        # This changes the output of compute action to a tuple (action, hidden_state, info)
        # We also pass in the previous hidden state in order for the model to correctly use the LSTM
        if MODE == 'test':
            action, hidden_state, _ = agent.compute_action(obs, state=hidden_state, full_fetch=True)
        else:
            action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    print('episode_reward', episode_reward)
    ray.shutdown()
    env.render()


if __name__ == '__main__':
    main()
