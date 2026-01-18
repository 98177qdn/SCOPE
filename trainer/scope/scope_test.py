import matplotlib.pyplot as plt
from model.scope import SCOPE
from envs.UAVEnv import UAVEnv
import numpy as np
from PIL import Image
import warnings
import io
import os

from trainer.scope.dynamics_model import MultiAgentDynamicModel

warnings.filterwarnings('ignore')

done_flag = False
final_frame = 0

################################### Testing ###################################
def test():
    n_actions = 2
    future_steps = 3
    seq_len = 5

    # === env initialize ===
    env = UAVEnv(dim_actions=n_actions)
    n_hunters = env.num_hunters

    # get dim of observation
    first_agent_key = list(env.hunters_observation_space.keys())[0]
    obs_dim = env.hunters_observation_space[first_agent_key].shape[0]
    preds_buffer = []

    # observation concatenate
    actor_dims = [obs_dim + future_steps * obs_dim for _ in range(n_hunters)]
    critic_dims = sum(actor_dims)

    # === initialize agent ===
    agents = SCOPE(actor_dims, critic_dims, n_hunters, n_actions, env,
                   fc1=128, fc2=128, alpha=0.0001, beta=0.002,
                   scenario='UAV_Round_up', chkpt_dir='../../tmp/scope/')

    agents.load_checkpoint()
    print('---- Evaluating ----')

    # === initialize SPDM ===
    SPDM = MultiAgentDynamicModel(n_hunters, seq_len, obs_dim, future_steps,
                                       chkpt_dir='../../tmp/scope/', name='DLinear.pt')
    SPDM.load_model()

    # === reset ===
    hunters_obs, _ = env.reset()
    rewards = np.zeros(env.num_hunters)
    obs_records = [hunters_obs]

    images = []
    save_dir = '../../result/scope/frames'
    os.makedirs(save_dir, exist_ok=True)
    total_steps = 0
    max_frames = 400
    done_flag = False

    for frame in range(max_frames):

        preds = SPDM.update_and_predict(hunters_obs)
        preds_buffer.append(preds)
        aug_obs = [
            np.concatenate([hunters_obs[i], preds[i].reshape(-1)])
            for i in range(n_hunters)
        ]

        hunters_actions, modified = agents.agents_choose_action(aug_obs, total_steps, True)
        targets_actions = env.targets_choose_action()


        hunters_next_obs, _, hunters_rewards, _, hunter_dones, safe_value = env.step(hunters_actions, targets_actions, modified)
        rewards += hunters_rewards

        step_mse = np.nan
        if len(preds_buffer) >= future_steps and len(obs_records) >= future_steps + 1:

            pred_block = np.array(list(preds_buffer[-future_steps].values()))

            true_block = np.stack([obs_records[-future_steps + t] for t in range(future_steps)])
            true_block = np.transpose(true_block, (1, 0, 2))

            diff = pred_block - true_block
            step_mse = float(np.mean(diff ** 2))

        print(f"Frame {frame} | Rewards: {str(rewards.tolist()):<90}| MSE:{step_mse}, cost:{np.sum(safe_value)}")

        env.render_anime_2D()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        images.append(img.copy())
        buf.close()
        img.save(os.path.join(save_dir, f'frame_{frame:03d}.png'))

        hunters_obs = hunters_next_obs
        obs_records.append(hunters_obs)

        total_steps += 1

        if any(hunter_dones):
            done_flag = True
            final_frame = frame
            print("Round-up finished in", frame, "steps.")
            break

    print("Collision has occurred ", env.collision_num, " times")

    if not os.path.exists('../../result/scope'):
        os.makedirs('../../result/scope')

    # save GIF
    if images:
        if done_flag:
            gif_name = '../../result/scope/roundup_success.gif'
        else:
            gif_name = '../../result/scope/roundup_fail.gif'

        images[0].save(gif_name, save_all=True, append_images=images[1:], duration=100, loop=0)
        print(f"GIF saved as '{gif_name}'")
    else:
        print("No images captured. GIF not saved.")


if __name__ == '__main__':
    test()