# %%
import warnings
import matplotlib.pyplot as plt
import numpy as np
import gymnasium
import torch
import torchrl
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import DoubleToFloat, TransformedEnv
import minari
from arc_agi_grid_env_coord import create_arc_env_coord, preprocess_data, cmap, \
                                    load_challenges_and_solutions, convert_int_to_dict, make_env
from pathlib import Path
from matplotlib import colors
# %%
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
training_challenges_json = "../datasets/arc-agi_training_challenges.json"
training_solutions_json = "../datasets/arc-agi_training_solutions.json"
evaluation_challenges_json = "../datasets/arc-agi_evaluation_challenges.json"
evaluation_solutions_json = "../datasets/arc-agi_evaluation_solutions.json"
test_challenges_json = "../datasets/arc-agi_test_challenges.json"

training_challenges, training_solutions, evaluation_challenges, \
evaluation_solutions, test_challenges = load_challenges_and_solutions(
                                        training_challenges_json,
                                        training_solutions_json,
                                        evaluation_challenges_json,
                                        evaluation_solutions_json,
                                        test_challenges_json,
                                    )
train_task_img_dict, _, train_img_shape_colors, _ = preprocess_data(training_challenges, training_solutions)
eval_task_img_dict, _, eval_img_shape_colors, _= preprocess_data(evaluation_challenges, evaluation_solutions)
# task_id = '3cd86f4f'
# pair_idx = 1
# print(img_shape_color_list[pair_idx])
# row = img_shape_color_list[pair_idx].row
# col = img_shape_color_list[pair_idx].col
# size_candidate = [row, col]
# color_candidate = img_shape_color_list[pair_idx].color
# %%
# pair_indices = list(range(len(train_task_img_dict[task_id])))
# pair_indices = [1,]
# pair_indices
# %%
env = make_env(training_challenges=training_challenges,
                training_solutions=training_solutions,
                evaluation_challenges=evaluation_challenges,
                evaluation_solutions=evaluation_solutions,
                test_challenges=test_challenges,
                train_task_img_dict=train_task_img_dict,
                eval_task_img_dict=eval_task_img_dict,
            )()
# %%
task_ids = list(train_task_img_dict.keys())[:10]
task_ids
# %%
minari.list_local_datasets()

# %%
# dataset_id = f"arc_agi/arc_agi_1000-v0"
dataset_id = f"arc_agi/arc_agi_101-v0"
# %%
buffer_dict = dict()
for task_id in task_ids:
    dataset_id = f"arc_agi/arc_agi_{task_id}-v0"
    dataset = minari.load_dataset(dataset_id, download=False)
    batch_size = 32
    replay_buffer = MinariExperienceReplay(
        dataset_id,
        download=False,
        load_from_local_minari=True,
        split_trajs=False,
        batch_size=batch_size,
        sampler=SamplerWithoutReplacement(),
        transform=DoubleToFloat(),
    )
    buffer_dict[task_id] = replay_buffer

# %%
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
warnings.simplefilter("ignore")
# %%
task_ids = list(train_task_img_dict.keys())[:10]
# %%
from torchrl.data.tensor_specs import OneHot
from tensordict.nn import TensorDictModule
from torchrl.modules.distributions.discrete import OneHotCategorical
from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.modules.tensordict_module.common import SafeModule
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tensordict import TensorDict
from torchrl.objectives import SoftUpdate
from torchrl.objectives.iql import DiscreteIQLLoss
from torchrl.trainers.helpers.models import ACTIVATIONS
from torchrl.data.tensor_specs import OneHot
from torch.nn.functional import one_hot
# %%
# from network.vit import ViTPolicy, ViTQValue, ViTValue

# n_act, n_obs = 9000, (30, 180)
# spec = OneHot(n_act)
# vit_policy = ViTPolicy(embed_dim=512, patch_size=15,
#                  num_heads=8, num_layers=12, action_size=9000,)
# actor_module = SafeModule(vit_policy, in_keys=["observation"], out_keys=["logits"])
# actor = ProbabilisticActor(
#             module=actor_module,
#             in_keys=["logits"],
#             out_keys=["action"],
#             spec=spec,
#             distribution_class=OneHotCategorical,
#             default_interaction_type=ExplorationType.DETERMINISTIC,
#             ).to(device)
# vit_q_value = ViTQValue(embed_dim=512, patch_size=15,
#                  num_heads=8, num_layers=12, action_size=9000)
# q_value_net = SafeModule(
#             vit_q_value,
#             in_keys=["observation"],
#             out_keys=["state_action_value"],
#             ).to(device)
# vit_value = ViTValue(embed_dim=512, patch_size=15,
#                  num_heads=8, num_layers=12, action_size=9000)
# value_net = SafeModule(
#             vit_value,
#             in_keys=["observation"],
#             out_keys=["state_value"],
#             ).to(device)

# %%
from network.cnn import CNNPolicy, CNNQValue, CNNValue

n_act, n_obs = 9000, (30, 180)
spec = OneHot(n_act)
cnn_policy = CNNPolicy(input_channels=1, hidden_dim=128, action_size=9000, dropout=0.1,)
actor_module = SafeModule(cnn_policy, in_keys=["observation"], out_keys=["logits"])
actor = ProbabilisticActor(
            module=actor_module,
            in_keys=["logits"],
            out_keys=["action"],
            spec=spec,
            distribution_class=OneHotCategorical,
            default_interaction_type=ExplorationType.DETERMINISTIC,
            ).to(device)
cnn_q_value = CNNQValue(input_channels=1, hidden_dim=128, action_size=9000, dropout=0.1)
q_value_net = SafeModule(
            cnn_q_value,
            in_keys=["observation"],
            out_keys=["state_action_value"],
            ).to(device)
cnn_value = CNNValue(input_channels=1, hidden_dim=128, dropout=0.1)
value_net = SafeModule(
            cnn_value,
            in_keys=["observation"],
            out_keys=["state_value"],
            ).to(device)


# %%
loss_module = DiscreteIQLLoss(actor, q_value_net, value_net,
                            loss_function="l2",
                            temperature=3,
                            expectile=0.7,)
loss_module.make_value_estimator(gamma=0.99)
target_net_updater = SoftUpdate(loss_module, tau=0.005)
optimizer = torch.optim.Adam(loss_module.parameters(), lr=0.0003)

# %%
def visualize_grid(iteration: int, env, task_id, pair_idx, w=0.5, first=False):
    """Visualize current grid state and log to wandb/tensorboard."""
    try:
        norm = colors.Normalize(vmin=0, vmax=11)
        # Get current grid from first environment
        info_dict = env._get_info()
        target_grid = info_dict['target_grid_img']
        current_grid = info_dict['current_grid_img']
        timestep = info_dict['timestep']
        task_id = info_dict['task_id']
        test_input_idx = info_dict['test_input_idx']
        
        test_sol_current_mat = current_grid[:, 120:]
        test_sol_target_mat = target_grid[:, 120:]
            
        if first:
            plt.imshow(test_sol_target_mat, cmap=cmap, norm=norm)
            plt.grid(True, which='both', color='lightgrey', linewidth=1.0)
            plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
            plt.grid(visible=True, which='both', color='#666666', linewidth=w)
            plt.xticks([x-0.5 for x in range(1 + len(test_sol_target_mat[0]))])
            plt.yticks([x-0.5 for x in range(1 + len(test_sol_target_mat))])
            plt.tick_params(axis='both', color='none', length=0)
            plt.title(f'task: #{task_id}  test_input_idx: #{test_input_idx}', fontsize=12, color='#000000')
            figure_folder_path = Path("./figures")
            if not figure_folder_path.exists():
                figure_folder_path.mkdir(parents=True)
            plt.savefig(f"./figures/{task_id}_{pair_idx}_target.png")
            plt.close()

        plt.imshow(test_sol_current_mat, cmap=cmap, norm=norm)
        plt.grid(True, which='both', color='lightgrey', linewidth=1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        plt.grid(visible=True, which='both', color='#666666', linewidth=w)
        plt.xticks([x-0.5 for x in range(1 + len(test_sol_current_mat[0]))])
        plt.yticks([x-0.5 for x in range(1 + len(test_sol_current_mat))])
        plt.tick_params(axis='both', color='none', length=0)
        plt.title(f'task: #{task_id}  test_input_idx: #{test_input_idx}  iter: #{iteration}  timestep: #{timestep}', 
                    fontsize=12, color='#000000')
        figure_folder_path = Path("./figures")
        if not figure_folder_path.exists():
            figure_folder_path.mkdir(parents=True)
        plt.savefig(f"./figures/{task_id}_{pair_idx}_{iteration}_current.png")
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not visualize grid at iteration {iteration}: {e}")

# %%
max_episode_steps = 900
@torch.no_grad()
def evaluate(iteration, env, policy, task_ids, visualize=False):
    num_sucess = 0
    episode_returns = []
    policy.eval()
    for task_id in task_ids:
        pair_indices = list(range(len(train_task_img_dict[task_id])))
        img_shape_color_list = train_img_shape_colors[task_id]
        for pair_idx in pair_indices:
            options = {'mode': 'train',
                    'task_id': task_id,
                    'pair_idx': pair_idx,
                    'rand_init': False,
                    'ratio_fill_correct': 0.0,
                    }
            row = img_shape_color_list[pair_idx].row
            col = img_shape_color_list[pair_idx].col
            size_candidate = [row, col]
            color_candidate = img_shape_color_list[pair_idx].color
            traj_len = size_candidate[0] * size_candidate[1]
            obs, info = env.reset(seed=seed, options=options)
            current_info = info
            done = False
            total_reward = 0
            t = 0
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action_logits = policy(obs_tensor)
                # print(action_logits)
                # print(action_logits[0].shape)
                # print(action_logits[1].shape)
                action = torch.argmax(action_logits[0], dim=1).item()
                # Convert integer action to dictionary format
                dict_action = convert_int_to_dict(action)
                # print(dict_action)
                obs, reward, terminated, truncated, info = env.step(action)
                # print(f"timestep {t}: {dict_action}, {reward}, {terminated}, {truncated}")
                t += 1
                total_reward += reward
                done = terminated or truncated
                if done:
                    success_reward = (((traj_len-1) * 0.05 + 1))
                    total_reward = np.round(total_reward, 2)
                    episode_returns.append(total_reward)
                    is_success = (total_reward == success_reward)
                    print(f"Task: {task_id}, Pair: {pair_idx}, Sucess: {is_success}, Total Reward: {total_reward}, Traj Length: {traj_len}")
                    if is_success:
                        num_sucess += 1
                    if visualize:
                        visualize_grid(iteration, env, task_id, pair_idx, first=True)
                    break
    policy.train()
    return num_sucess, episode_returns
# %%
from tqdm.auto import tqdm
iterations = 50000  # Set to 50_000 to reproduce the results below
eval_interval = 100
num_eval_episodes = 1
total_loss_logs = []
task_loss_logs_dict = dict()
for task_id in task_ids:
    task_loss_logs_dict[task_id] = []
eval_reward_logs = []
pbar = tqdm(range(iterations))
max_num_success = -1
# for i in pbar:
for i in range(iterations):
    # 1) Sample data from the dataset
    total_loss = 0
    for task_id in task_ids:
        replay_buffer = buffer_dict[task_id]
        data = replay_buffer.sample().cuda()
        # print(data['observation'].shape)
        data['action'] = one_hot(data['action'], num_classes=9000)
        # 2) Compute loss l = L_V + L_Q + L_pi
        loss_dict = loss_module(data.to(device))
        loss = loss_dict["loss_value"] + loss_dict["loss_qvalue"] + loss_dict["loss_actor"]
        task_loss_logs_dict[task_id].append(loss)
        total_loss += loss
    total_loss_logs.append(total_loss.item())
    # 3) Backpropagate the gradients
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()  # Update V(s), Q(a, s), pi(a|s)
    target_net_updater.step()  # Update the target Q-network

    # Evaluate the policy
    if i % eval_interval == 0:
        num_success, episode_returns = evaluate(i, env, actor, task_ids, visualize=True)
        pbar.set_description(
            f"Loss: {total_loss_logs[-1]:.1f}, Num Sucess: {num_success} Avg return: {np.mean(episode_returns):.2f}"
        )
        eval_reward_logs.append(np.mean(episode_returns))
        print(f"Epoch {i}: Loss: {total_loss_logs[-1]:.1f}, Num Sucess: {num_success} Avg return: {np.mean(episode_returns):.2f}")
    
    # if num_success > max_num_success:
    #     num_success, episode_returns = evaluate(i, env, actor, task_ids, visualize=True)
    #     max_num_success = num_success
    print(f"Epoch {i}: Loss: {total_loss_logs[-1]:.4f}")
    
pbar.close()

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
axes[0].plot(total_loss_logs)
axes[0].set_title("Loss")
axes[0].set_xlabel("iterations")
axes[0].set_ylim(0, 500)
axes[1].plot(eval_reward_logs)
axes[1].set_title("Cumulative reward")
axes[1].set_xlabel("iterations")
fig.tight_layout()
plt.show()
# %%
num_success, episode_returns = evaluate(i, env, actor, pair_indices)

# %%
print(loss)