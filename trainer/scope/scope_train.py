import os
import warnings
import numpy as np
from PIL import Image

from model.scope import SCOPE
from envs.UAVEnv import UAVEnv
from memory.memory import ReplayBuffer
from dynamics_model import MultiAgentDynamicModel

warnings.filterwarnings('ignore')

def obs_list_to_state_vector(obs):
    all_state = np.hstack([np.ravel(o) for o in obs])
    return all_state

def save_image(env_render, filename):
    image = Image.fromarray(env_render, 'RGBA')
    image = image.convert('RGB')
    image.save(filename)

################################### Training ###################################
def train():
    ####### initialize environment hyperparameters ######
    n_actions = 2
    env = UAVEnv(dim_actions=n_actions)
    n_hunters = env.num_hunters
    env_name = 'UAV_Round_up'

    # === config ===
    PRINT_INTERVAL = 10
    N_GAMES = 10001
    MAX_STEPS = 100
    total_steps = 0
    score_history = []
    cost_history = []
    best_score = -3000
    train_interval = 10
    predictor_train_interval = 10        # train interval of predictor
    warmup_steps = 100                   # warmup steps of predictor
    use_predictor = True                 # use predictor or not
    #####################################################

    ################## initialize model #################

    # === initialize predictor ===
    predictor = None
    if use_predictor:
        future_steps = 3  # predict steps
        seq_len = 5  # input steps of predictor
        first_agent_key = list(env.hunters_observation_space.keys())[0]
        obs_dim = env.hunters_observation_space[first_agent_key].shape[0]
        actor_dims = [obs_dim + future_steps * obs_dim for _ in range(n_hunters)]  # raw observation + predict observation
        critic_dims = sum(actor_dims)

        predictor = MultiAgentDynamicModel(
            n_agents=n_hunters,
            seq_len=seq_len,
            obs_dim=obs_dim,
            pred_len=future_steps,
            chkpt_dir='../../tmp/scope/',
            name='DLinear.pt'
        )
    else:
        actor_dims = [env.hunters_observation_space[agent_id].shape[0] for agent_id in env.hunters_observation_space]
        critic_dims = sum(actor_dims)

    # === initialize agents ===
    agents = SCOPE(
        actor_dims, critic_dims, n_hunters, n_actions, env,
        fc1=128, fc2=128,
        alpha=0.0001, beta=0.002, gamma=0.99, tau=0.01,
        scenario=env_name, chkpt_dir='../../tmp/scope/'
    )

    # === initialize replay buffer ===
    memory = ReplayBuffer(
        max_size=1_000_000,
        critic_dims=critic_dims,
        actor_dims=actor_dims,
        n_actions=n_actions,
        n_agents=n_hunters,
        batch_size=256
    )
    #####################################################

    ###################### logging ######################

    # === log files for multiple runs are NOT overwritten ===
    log_dir = "../../train_logs/scope"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # === get number of log files in log directory ===
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    # === create new log file for each run ===
    log_f_name = log_dir + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    # === logging file ===
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward,cost,SPDM\n')
    #####################################################

    ################# training procedure ################
    print('---- training start ----')

    # === training loop ===
    for i_episode in range(N_GAMES):
        # === environment reset ===
        raw_hunters_obs, _ = env.reset()  # raw observation

        if use_predictor:
            # === predict states concatenate ===
            preds = predictor.update_and_predict(raw_hunters_obs)
            hunters_obs = [np.concatenate([raw_hunters_obs[j], preds[j].reshape(-1)]) for j in range(n_hunters)]
        else:
            hunters_obs = raw_hunters_obs

        score = 0
        cost = 0
        hunter_dones = [False] * n_hunters
        episode_step = 0
        predict_losses = []

        while not any(hunter_dones):
            # === hunters choose actions ===
            hunters_actions, modified = agents.agents_choose_action(hunters_obs, total_steps, False)

            # === targets choose actions ===
            targets_actions = env.targets_choose_action()

            # === Step ===
            raw_hunters_obs_next, _, hunters_rewards, _, hunter_dones, safe_values = env.step(hunters_actions, targets_actions, modified)

            if use_predictor:
                # === predict states concatenate ===
                preds_next = predictor.update_and_predict(raw_hunters_obs_next)
                hunters_next_obs = [np.concatenate([raw_hunters_obs_next[j], preds_next[j].reshape(-1)]) for j in range(n_hunters)]
            else:
                hunters_next_obs = raw_hunters_obs_next

            # === store experience ===
            state = obs_list_to_state_vector(hunters_obs)
            state_ = obs_list_to_state_vector(hunters_next_obs)

            if episode_step >= MAX_STEPS:
                hunter_dones = [True] * n_hunters

            memory.store_transition(
                hunters_obs, state, hunters_actions, hunters_rewards,
                hunters_next_obs, state_, hunter_dones, episode_step, safe_values
            )

            # === learning ===
            if total_steps % train_interval == 0:
                agents.learn(memory)

            # === predictor learning ===
            if total_steps > warmup_steps and total_steps % predictor_train_interval == 0 and use_predictor:
                predict_loss = predictor.train_step(memory)
                predict_losses.append(predict_loss)

            # === update states ===
            hunters_obs = hunters_next_obs
            score += sum(hunters_rewards)
            cost += sum(safe_values)
            total_steps += 1
            episode_step += 1

        # === log in logging file ===
        score_history.append(score.item())
        cost_history.append(cost.item())
        avg_score = np.mean(score_history[-PRINT_INTERVAL:])
        avg_cost = np.mean(cost_history[-PRINT_INTERVAL:])
        avg_predict_loss = np.mean(predict_losses)

        if i_episode % PRINT_INTERVAL == 0 and i_episode > 0 and avg_score > best_score:
            print('New best score', avg_score, '>', best_score, 'saving models...')
            agents.save_checkpoint()

            if use_predictor:
                predictor.save_model()
            best_score = avg_score

        if i_episode % PRINT_INTERVAL == 0 and i_episode > 0:
            print(f'episode {i_episode}, average score {avg_score:.1f}, cost {avg_cost:.1f}, predicted loss {avg_predict_loss:.3f}')
            agents.save_checkpoint('../../tmp/scope/last/')
            if use_predictor:
                predictor.save_model('../../tmp/scope/last/')

            # === log average reward till last episode ===
            log_avg_reward = round(avg_score, 4)
            log_avg_cost = round(avg_cost, 4)
            log_avg_predict_loss = round(avg_predict_loss, 4)
            log_f.write('{},{},{},{},{}\n'.format(i_episode, total_steps, log_avg_reward, log_avg_cost, log_avg_predict_loss))
            log_f.flush()

    #####################################################

if __name__ == '__main__':
    train()