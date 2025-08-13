# %%
from typing import Optional
import numpy as np
import gymnasium as gym
from itertools import permutations, product
import json
from typing import Tuple, Dict, Union, List
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap, Normalize
import plotly.graph_objects as go
import plotly.express as px

cmap = colors.ListedColormap(
    ['#000000', # 0: black
     '#0074D9', # 1: blue
     '#FF4136', # 2: red
     '#2ECC40', # 3: green
     '#FFDC00', # 4: yello
     '#AAAAAA', # 5: gray
     '#F012BE', # 6: magenta
     '#FF851B', # 7: oragne
     '#7FDBFF', # 8: sky
     '#870C25', # 9: brwon
     '#FFFFFF', # 10: mask
     ])
norm = colors.Normalize(vmin=0, vmax=10)


def preprocess_data(challenges,
                    solutions
                    ) -> Tuple[Dict, Dict]:
    # TODO 1: Padding 수행
    # TODO 2: train_pairs, test_pair 합쳐서 30x660 이미지 데이터 만들기
    # TODO 3: 30 x 660 데이터 30x30 단위로 flatten 해서 19800 시퀀스 만들기
    # TODO 4: 이미지, 시퀀스 데이터들 리턴

    max_input_shape = (30, 30)
    max_output_shape = (30, 30)
    pad_val = 10
    task_img_dict = dict()
    task_seq_dict = dict()

    for (task_id, task_data), (task_id, task_sol) in zip(challenges.items(), solutions.items()):
        # task 내의 train pair 들의 최대 개수는 10
        train_task_img_pairs = []
        train_task_seq_pairs = []
        for pair in task_data.get('train', []):
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            # Append input and output grids to the respective lists
            padded_input = np.pad(input_grid, [(0, max_input_shape[0] - input_grid.shape[0]), (0, max_input_shape[1] - input_grid.shape[1])], mode='constant', constant_values=pad_val)
            padded_output = np.pad(output_grid, [(0, max_output_shape[0] - output_grid.shape[0]), (0, max_output_shape[1] - output_grid.shape[1])], mode='constant', constant_values=pad_val)
            img_XY = np.concatenate([padded_input, padded_output], axis=-1)
            train_task_img_pairs.append(img_XY)
            seq_input = padded_input.flatten()
            seq_output = padded_output.flatten()
            seq_XY = np.concatenate([seq_input, seq_output], axis=-1)
            train_task_seq_pairs.append(seq_XY)
        if len(train_task_img_pairs) < 10:
            dummy_img = pad_val * np.ones([30, 60])
            dummy_seq = pad_val * np.ones([30 * 60,])
            while len(train_task_img_pairs) < 10:
                train_task_img_pairs.append(dummy_img)
                train_task_seq_pairs.append(dummy_seq)
        ary_train_img_pairs = np.hstack(train_task_img_pairs)
        ary_train_seq_pairs = np.concatenate(train_task_seq_pairs)
        test_task_img_pairs = []
        test_task_seq_pairs = []
        for test_pair, test_sol in zip(task_data.get('test', []), task_sol):
            input_grid = np.array(test_pair['input'])
            output_grid = np.array(test_sol)
            padded_input = np.pad(input_grid, [(0, max_input_shape[0] - input_grid.shape[0]), (0, max_input_shape[1] - input_grid.shape[1])], mode='constant', constant_values=pad_val)
            padded_output = np.pad(output_grid, [(0, max_output_shape[0] - output_grid.shape[0]), (0, max_output_shape[1] - output_grid.shape[1])], mode='constant', constant_values=pad_val)
            img_XY = np.concatenate([padded_input, padded_output], axis=-1)
            test_task_img_pairs.append(img_XY)
            seq_input = padded_input.flatten()
            seq_output = padded_output.flatten()
            seq_XY = np.concatenate([seq_input, seq_output], axis=-1)
            test_task_seq_pairs.append(seq_XY)
        train_test_img_pair_list = []
        train_test_seq_pair_list = []
        for (test_img_pair, train_seq_pair) in zip(test_task_img_pairs, test_task_seq_pairs):
            train_test_img_pair = np.hstack([ary_train_img_pairs, np.array(test_img_pair)])
            train_test_img_pair_list.append(train_test_img_pair)
            train_test_seq_pair= np.concatenate([ary_train_seq_pairs, np.array(train_seq_pair)])
            train_test_seq_pair_list.append(train_test_seq_pair)
        task_img_dict[task_id] = train_test_img_pair_list
        task_seq_dict[task_id] =train_test_seq_pair_list
    return task_img_dict, task_seq_dict


class ArcAgiGridEnv(gym.Env):
    def __init__(self,
                 training_challenges_json: str,
                 training_solutions_json: str,
                 evaluation_challenges_json: str,
                 evaluation_solutions_json: str,
                 test_challenges_json: str,
                 ):
        # training, evaluation, test challenge 및 solution들을 불러오기
        with open(training_challenges_json, 'r', encoding='utf-8') as file:
            self.training_challenges = json.load(file)
        with open(training_solutions_json, 'r', encoding='utf-8') as file:
            self.training_solutions = json.load(file)
        with open(evaluation_challenges_json, 'r', encoding='utf-8') as file:
            self.evaluation_challenges = json.load(file)
        with open(evaluation_solutions_json, 'r', encoding='utf-8') as file:
            self.evaluation_solutions = json.load(file)
        with open(test_challenges_json, 'r', encoding='utf-8') as file:
            self.test_challenges = json.load(file)
        self.train_task_img_dict, self.train_task_seq_dict = preprocess_data(self.training_challenges, self.training_solutions)
        self.eval_task_img_dict, self.eval_task_seq_dict = preprocess_data(self.evaluation_challenges, self.evaluation_solutions)
        self.train_task_list = list(self.train_task_img_dict.keys())
        self.eval_task_list = list(self.eval_task_img_dict.keys())

        # observation space에 대한 정의
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "grid_img": gym.spaces.Box(low=0, high=10, shape=(30,660), dtype=int),
                "grid_seq": gym.spaces.Box(low=0, high=10, shape=(19800,), dtype=int),
            }
        )
        # action space에 대한 정의 (0~9 색상, 10: 마스크)
        self.action_space = gym.spaces.Discrete(11)

    def _select_task(self, seed) -> str:
        random.seed(seed)
        np.random.seed(seed)
        task_id = random.choice(self.train_task_list)
        return task_id

    def _get_reward(self,
                    ) -> float:
        """
        positive reward: 현재 timestep까지 맞춘 영역 수 * 1/900
        negative reward: 현재 timestep까지 틀린 영역 수 * -1/900
        gemetric reward: 그림의 패턴을 고려한 보상 (구현예정)
        """
        target_grid_test_sol: int = self._target_grid_img[:, 630:]
        current_grid_test_sol: int = self._current_grid_img[:, 630:]
        target_grid_test_sol_cell: int = self._target_grid_seq[18900+self.timestep]
        current_grid_test_sol_cell: int = self._current_grid_seq[18900+self.timestep]
        target_grid_test_sol_seq = self._target_grid_seq[18900:18900+self.timestep+1]
        current_grid_test_sol_seq = self._current_grid_seq[18900:18900+self.timestep+1]
        num_correct_cells: int = np.sum((target_grid_test_sol_seq == current_grid_test_sol_seq).astype(int))
        positive_reward: float = 1/900 * num_correct_cells
        num_wrong_cells: int = np.sum((target_grid_test_sol_seq != current_grid_test_sol_seq).astype(int))
        negative_reward: float = -1/900 * num_wrong_cells
        geometric_reward = 0
        return positive_reward + + negative_reward + geometric_reward

    def _get_obs(self) -> Dict:
        return {"current_grid_img": self._current_grid_img,
                "current_grid_seq": self._current_grid_seq}

    def _get_info(self) -> Dict:
        return {
            'target_grid_img': self._target_grid_img,
            'target_grid_seq': self._target_grid_seq,
            'timestep': self.timestep,
            'task_id': self.task_id,
            'test_input_idx': self.test_input_idx,
        }

    def reset(self,
              seed: Optional[int] = None,
              mode: str = 'train',
              task_id = None,
              reset_sol_grid: str = 'padding',
              options: Optional[dict] = None):
        self.timestep = 0
        if task_id == None:
            task_id = self._select_task(seed)
        self.task_id = task_id
        """
        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)
            mode: (train, evaluation, test)
        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        # task_id에 해당하는 target grid 선택 (test input이 여러 개 존재 가능하므로 한 번 더 random.choice 수행
        if mode == 'train':
            test_input_idx = random.choice(list(range(len(self.train_task_img_dict[task_id]))))
            self._target_grid_img = self.train_task_img_dict[task_id][test_input_idx]
            self._target_grid_seq = self.train_task_seq_dict[task_id][test_input_idx]
        elif mode == 'evaluation' or mode == 'eval':
            test_input_idx = random.choice(list(range(len(self.eval_task_img_dict[task_id]))))
            self._target_grid_img = self.eval_task_img_dict[task_id][test_input_idx]
            self._target_grid_seq = self.eval_task_seq_dict[task_id][test_input_idx]
        self.test_input_idx = test_input_idx
        # target grid에서 test solution에 해당하는 부분을 전부 pad_val으로 masking하고 current grid로 할당
        if reset_sol_grid == 'padding':
            pad_val= 10
            self._current_grid_img = self._target_grid_img.copy()
            self._current_grid_img[0:30, 630:] = pad_val # H: 0:30 W: 630:660
            self._current_grid_seq = self._target_grid_seq.copy()
            self._current_grid_seq[18900:] = pad_val
        elif reset_sol_grid == 'random':
            # ! 여기서 solution에 해당하는 부분을 랜덤으로 초기화해도 좋을듯?
            rand_grid = np.random.randint(low=0,
                                            high=10,
                                            size=(30, 30))
            self._current_grid_img = self._target_grid_img.copy()
            self._current_grid_img[0:30, 630:] = rand_grid
            self._current_grid_seq = self._target_grid_seq.copy()
            self._current_grid_seq[18900:] = rand_grid.flatten()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: Union[int]):
        """Execute one timestep within the environment.
        Args:
            action: The action to take (0-10)
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        row = self.timestep // 30
        col = self.timestep % 30
        self._current_grid_img[row, 630+col] = action # H: 0:30 W: 630:660
        self._current_grid_seq[18900+self.timestep] = action

        # Check if agent reached the target
        terminated = (self.timestep == 900)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()
        self.timestep += 1
        return observation, reward, terminated, truncated, info

    def plot_current_task_and_sol(self, mode='train'):
        self.plot_one_task(mode, self.task_id)
        
    def plot_one_task(self, mode, task_id, size=2.5, w1=0.9):
        if mode == 'train':
            task = self.training_challenges[task_id]
            task_solutions = self.training_solutions[task_id]
        elif mode == 'evaluation' or self.mode == 'eval':
            task = self.evaluation_challenges[task_id]
            task_solutions = self.evaluation_solutions[task_id]
        else:
            raise NotImplementedError
        titleSize=16
        num_train = len(task['train'])
        num_test  = len(task['test'])
        wn=num_train+num_test
        fig, axs  = plt.subplots(2, wn, figsize=(size*wn,2*size))
        plt.suptitle(f'Task #{task_id}', fontsize=titleSize, fontweight='bold', y=1, color = '#eeeeee')
        '''train:'''
        for j in range(num_train):
            self.plot_one(axs[0, j], j,task, 'train', 'input',  w=w1)
            self.plot_one(axs[1, j], j,task, 'train', 'output', w=w1)
        '''test:'''
        for k in range(num_test):
            self.plot_one(axs[0, j+k+1], k, task, 'test', 'input', w=w1)
            task['test'][k]['output'] = task_solutions[k]
            self.plot_one(axs[1, j+k+1], k, task, 'test', 'output', w=w1)
        axs[1, j+1].set_xticklabels([])
        axs[1, j+1].set_yticklabels([])
        axs[1, j+1] = plt.figure(1).add_subplot(111)
        axs[1, j+1].set_xlim([0, wn])
        '''Separators:'''
        colorSeparator = 'white'
        for m in range(1, wn):
            axs[1, j+1].plot([m,m],[0,1],'--', linewidth=1, color = colorSeparator)
        axs[1, j+1].plot([num_train,num_train],[0,1],'-', linewidth=3, color = colorSeparator)
        axs[1, j+1].axis("off")
        '''Frame and background:'''
        fig.patch.set_linewidth(5) #widthframe
        fig.patch.set_edgecolor('black') #colorframe
        fig.patch.set_facecolor('#444444') #background
        plt.tight_layout()
        print(f'#{task_id}') # for fast and convinience search
        plt.show()

    def plot_one(self, ax, i, task, train_or_test, input_or_output, solution=None, w=0.8):
        fs=12
        input_matrix = task[train_or_test][i][input_or_output]
        ax.imshow(input_matrix, cmap=cmap, norm=norm)
        ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
        ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
        '''Grid:'''
        ax.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
        ax.tick_params(axis='both', color='none', length=0)
        '''sub title:'''
        ax.set_title(train_or_test + ' ' + input_or_output, fontsize=fs, color = '#dddddd')

    def plot_original_task(self, task_id, train_or_test, i, input_or_output, mode='train', w=0.8):
        fs=12
        if mode == 'train':
            task = self.training_challenges[task_id]
        elif mode == 'evaluation' or self.mode == 'eval':
            task = self.evaluation_challenges[task_id]
        input_matrix = task[train_or_test][i][input_or_output]
        plt.imshow(input_matrix, cmap=cmap, norm=norm)
        plt.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        '''Grid:'''
        plt.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
        plt.xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
        plt.yticks([x-0.5 for x in range(1 + len(input_matrix))])
        plt.tick_params(axis='both', color='none', length=0)
        '''sub title:'''
        plt.title(f'task: {task_id}' + '  ' + train_or_test + ' ' + input_or_output + f'  #{i}', fontsize=fs, color = '#000000')

    def plot_padded_task(self, task_id, i, w=0.5):
        fs=12
        task = self.train_task_img_dict[task_id]
        input_matrix = task[i]
        plt.figure(figsize=(200, 500)) #
        plt.imshow(input_matrix, cmap=cmap, norm=norm)
        plt.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        # '''Grid:'''
        plt.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
        plt.xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
        plt.yticks([x-0.5 for x in range(1 + len(input_matrix))])
        plt.tick_params(axis='both', color='none', length=0)
        '''sub title:'''
        plt.title(f'task: {task_id}' + '   ' + f'#{i}', fontsize=fs, color = '#000000')

    def plot_current_grid(self, w=0.5):
        fs=12
        test_sol_current_mat = self._current_grid_img[:, 630:]
        plt.imshow(test_sol_current_mat, cmap=cmap, norm=norm)
        plt.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        # '''Grid:'''
        plt.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
        plt.xticks([x-0.5 for x in range(1 + len(test_sol_current_mat[0]))])
        plt.yticks([x-0.5 for x in range(1 + len(test_sol_current_mat))])
        plt.tick_params(axis='both', color='none', length=0)
        '''sub title:'''
        plt.title(f'task: {self.task_id}' + '   ' + f'#{self.test_input_idx}', fontsize=fs, color = '#000000')

    def plot_target_grid(self, w=0.5):
        fs=12
        test_sol_target_mat = self._target_grid_img[:, 630:]
        plt.imshow(test_sol_target_mat, cmap=cmap, norm=norm)
        plt.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        # '''Grid:'''
        plt.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
        plt.xticks([x-0.5 for x in range(1 + len(test_sol_target_mat[0]))])
        plt.yticks([x-0.5 for x in range(1 + len(test_sol_target_mat))])
        plt.tick_params(axis='both', color='none', length=0)
        '''sub title:'''
        plt.title(f'task: {self.task_id}' + '   ' + f'#{self.test_input_idx}', fontsize=fs, color = '#000000')

# %%
env = ArcAgiGridEnv(
    training_challenges_json='../datasets/arc-agi_training_challenges.json',
    training_solutions_json='../datasets/arc-agi_training_solutions.json',
    evaluation_challenges_json='../datasets/arc-agi_evaluation_challenges.json',
    evaluation_solutions_json='../datasets/arc-agi_evaluation_solutions.json',
    test_challenges_json='../datasets/arc-agi_test_challenges.json',
    )
# 794b24be train input pair 10개
# 8dab14c2 test input 4개
# 3cd86f4f
obs, info = env.reset(seed=1,
                        mode='train',
                        task_id='3cd86f4f',
                        reset_sol_grid='random')
# %%
env.plot_current_task_and_sol()
# %%
# 264363fd
env.plot_padded_task(task_id='3cd86f4f', i=0)

# %%
env.plot_current_grid()
# %%
env.plot_target_grid()

# %%
env.plot_one_task(mode='train', 
                  task_id='794b24be')
# %%
env.plot_original_task(task_id='3cd86f4f',
             train_or_test='test',
             i=2,
             input_or_output='output',
             )

# %%
obs, info = env.reset(seed=12,
                        mode='train',
                        # task_id='8dab14c2',
                        reset_sol_grid='random')
test_sol = env._target_grid_seq[18900:]
total_reward = 0
for t in range(900):
    action = test_sol[t]
    if t > 800:
        action = env.action_space.sample()
    next_obs, reward, terminated, trucated, info = env.step(action)
    total_reward += reward
    print(reward)
env.plot_current_grid()
print(total_reward)
# %%
obs, info = env.reset(seed=12,
                        mode='train',
                        # task_id='8dab14c2',
                        reset_sol_grid='random')
test_sol = env._target_grid_seq[18900:]
total_reward = 0
for t in range(900):
    action = env.action_space.sample()
    next_obs, reward, terminated, trucated, info = env.step(action)
    total_reward += reward
    print(reward)
env.plot_current_grid()
print(total_reward)
# %%
