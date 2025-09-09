# %%
from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from itertools import permutations, product
import json
from typing import Tuple, Dict, Union, List, Any
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap, Normalize

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


def preprocess_data(challenges: Dict[str, Any], solutions: Dict[str, Any]) -> Tuple[Dict[str, List], Dict[str, List]]:
    """
    Optimized preprocessing function for ARC AGI 2 dataset.

    Args:
        challenges: Dictionary containing challenge data
        solutions: Dictionary containing solution data

    Returns:
        Tuple of (dict_XYXYXY_img_pairs, dict_XYXYXY_seq_pairs)
    """
    MAX_SHAPE = (30, 30)
    PAD_VAL = 10

    dict_XYXYXY_img_pairs = {}
    dict_XYXYXY_seq_pairs = {}

    # Process each task
    for task_id, task_data in challenges.items():
        task_sol = solutions[task_id]

        # Process training pairs
        train_pairs_img, train_pairs_seq = _process_pairs(
            task_data.get('train', []),
            MAX_SHAPE,
            PAD_VAL
        )

        # Process test pairs
        test_inputs = task_data.get('test', [])
        test_pairs_img, test_pairs_seq = _process_test_pairs(
            test_inputs,
            task_sol,
            MAX_SHAPE,
            PAD_VAL
        )

        # Generate XYXYXY pairs efficiently
        dict_XYXYXY_img_pairs[task_id] = _generate_xyxyxy_pairs(
            train_pairs_img, test_pairs_img, is_sequence=False
        )

        dict_XYXYXY_seq_pairs[task_id] = _generate_xyxyxy_pairs(
            train_pairs_seq, test_pairs_seq, is_sequence=True
        )

    return dict_XYXYXY_img_pairs, dict_XYXYXY_seq_pairs


def _pad_grid(grid: np.ndarray, target_shape: Tuple[int, int], pad_val: int) -> np.ndarray:
    """Efficiently pad a grid to target shape."""
    pad_height = max(0, target_shape[0] - grid.shape[0])
    pad_width = max(0, target_shape[1] - grid.shape[1])

    if pad_height == 0 and pad_width == 0:
        return grid

    return np.pad(grid, [(0, pad_height), (0, pad_width)],
                  mode='constant', constant_values=pad_val)


def _process_pairs(pairs: List[Dict], max_shape: Tuple[int, int], pad_val: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Process input-output pairs into both image and sequence formats."""
    img_pairs = []
    seq_pairs = []

    for pair in pairs:
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])

        # Pad grids
        padded_input = _pad_grid(input_grid, max_shape, pad_val)
        padded_output = _pad_grid(output_grid, max_shape, pad_val)

        # Image format: concatenate along width (axis=1)
        xy_img = np.concatenate([padded_input, padded_output], axis=1)
        img_pairs.append(xy_img)

        # Sequence format: flatten and concatenate
        seq_input = padded_input.flatten()
        seq_output = padded_output.flatten()
        xy_seq = np.concatenate([seq_input, seq_output])
        seq_pairs.append(xy_seq)

    return img_pairs, seq_pairs


def _process_test_pairs(test_inputs: List[Dict], solutions: List, max_shape: Tuple[int, int], pad_val: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Process test pairs with their solutions."""
    img_pairs = []
    seq_pairs = []

    for test_input, solution in zip(test_inputs, solutions):
        input_grid = np.array(test_input['input'])
        output_grid = np.array(solution)

        # Pad grids
        padded_input = _pad_grid(input_grid, max_shape, pad_val)
        padded_output = _pad_grid(output_grid, max_shape, pad_val)

        # Image format
        xy_img = np.concatenate([padded_input, padded_output], axis=1)
        img_pairs.append(xy_img)

        # Sequence format
        seq_input = padded_input.flatten()
        seq_output = padded_output.flatten()
        xy_seq = np.concatenate([seq_input, seq_output])
        seq_pairs.append(xy_seq)

    return img_pairs, seq_pairs


def _generate_xyxyxy_pairs(train_pairs: List[np.ndarray], test_pairs: List[np.ndarray], is_sequence: bool) -> List[np.ndarray]:
    """Generate XYXYXY pairs from training and test data."""
    if not train_pairs or not test_pairs:
        return []

    # Generate all training pair permutations (XYXY format)
    train_xyxy_pairs = []
    for p in permutations(train_pairs, 2):
        if is_sequence:
            xyxy_pair = np.concatenate(p)
        else:
            xyxy_pair = np.hstack(p)
        train_xyxy_pairs.append(xyxy_pair)

    # Generate XYXYXY combinations
    xyxyxy_pairs = []
    for train_pair, test_pair in product(train_xyxy_pairs, test_pairs):
        if is_sequence:
            xyxyxy_pair = np.concatenate([train_pair, test_pair])
        else:
            xyxyxy_pair = np.hstack([train_pair, test_pair])
        xyxyxy_pairs.append(xyxyxy_pair)

    return xyxyxy_pairs


# Memory-efficient version for large datasets
def preprocess_data_generator(challenges: Dict[str, Any], solutions: Dict[str, Any]):
    """
    Generator version that yields one task at a time to reduce memory usage.

    Yields:
        Tuple of (task_id, img_pairs, seq_pairs)
    """
    MAX_SHAPE = (30, 30)
    PAD_VAL = 10

    for task_id, task_data in challenges.items():
        task_sol = solutions[task_id]

        # Process training pairs
        train_pairs_img, train_pairs_seq = _process_pairs(
            task_data.get('train', []), MAX_SHAPE, PAD_VAL
        )

        # Process test pairs
        test_inputs = task_data.get('test', [])
        test_pairs_img, test_pairs_seq = _process_test_pairs(
            test_inputs, task_sol, MAX_SHAPE, PAD_VAL
        )

        # Generate XYXYXY pairs
        img_pairs = _generate_xyxyxy_pairs(train_pairs_img, test_pairs_img, is_sequence=False)
        seq_pairs = _generate_xyxyxy_pairs(train_pairs_seq, test_pairs_seq, is_sequence=True)

        yield task_id, img_pairs, seq_pairs

def plot_one_task(challenges, solutions, task_id, size=2.5, w1=0.9):
    task = challenges[task_id]
    task_solutions = solutions[task_id]
    titleSize=16
    num_train = len(task['train'])
    num_test  = len(task['test'])
    wn=num_train+num_test
    fig, axs  = plt.subplots(2, wn, figsize=(size*wn,2*size))
    plt.suptitle(f'Task #{task_id}', fontsize=titleSize, fontweight='bold', y=1, color = '#eeeeee')
    '''train:'''
    for j in range(num_train):
        plot_one(axs[0, j], j,task, 'train', 'input',  w=w1)
        plot_one(axs[1, j], j,task, 'train', 'output', w=w1)
    '''test:'''
    for k in range(num_test):
        plot_one(axs[0, j+k+1], k, task, 'test', 'input', w=w1)
        task['test'][k]['output'] = task_solutions[k]
        plot_one(axs[1, j+k+1], k, task, 'test', 'output', w=w1)
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


def plot_one(ax, i, task, train_or_test, input_or_output, solution=None, w=0.8):
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


def plot_padded_task(train_task_img_dict, task_id, i, w=0.5):
    fs=12
    task = train_task_img_dict[task_id]
    input_matrix = task[i]
    plt.figure(figsize=(100, 200)) #
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


# %%
training_challenges_json = '../datasets/arc-agi_training_challenges.json'
training_solutions_json = '../datasets/arc-agi_training_solutions.json'
evaluation_challenges_json = '../datasets/arc-agi_evaluation_challenges.json'
evaluation_solutions_json = '../datasets/arc-agi_evaluation_solutions.json'
test_challenges_json = None
with open(training_challenges_json, 'r', encoding='utf-8') as file:
    training_challenges = json.load(file)
with open(training_solutions_json, 'r', encoding='utf-8') as file:
    training_solutions = json.load(file)
with open(evaluation_challenges_json, 'r', encoding='utf-8') as file:
    evaluation_challenges = json.load(file)
with open(evaluation_solutions_json, 'r', encoding='utf-8') as file:
    evaluation_solutions = json.load(file)
if test_challenges_json is not None:
    with open(test_challenges_json, 'r', encoding='utf-8') as file:
        test_challenges = json.load(file)
        
train_task_img_dict, train_task_seq_dict = preprocess_data(training_challenges, training_solutions)
eval_task_img_dict, eval_task_seq_dict = preprocess_data(evaluation_challenges, evaluation_solutions)
train_task_list = list(train_task_img_dict.keys())
eval_task_list = list(eval_task_img_dict.keys())

len(train_task_img_dict['007bbfb7'])
train_task_img_dict['007bbfb7']
plot_one_task(training_challenges, training_solutions, '007bbfb7')
plot_padded_task(train_task_img_dict, '007bbfb7', 0)



# %%
from torch.utils.data import Dataset, DataLoader
class ARCAGIDataset(Dataset):
    def __init__(self, 
                challenge_json_path: str,
                solution_json_path: str,
                task_id = None,
                ):
        super().__init__()
        with open(challenge_json_path, 'r', encoding='utf-8') as file:
            challenges = json.load(file)
        with open(solution_json_path, 'r', encoding='utf-8') as file:
            solutions = json.load(file)
        task_img_dict, task_seq_dict = preprocess_data(challenges, solutions)
        del task_seq_dict
        if task_id is None:
            xyxyxy_pairs = []
            for task_id in task_img_dict.keys():
                xyxyxy_pairs += task_img_dict[task_id]
            self.XYXYXY_pairs = torch.tensor(np.stack(xyxyxy_pairs)).unsqueeze(1)
        else:
            self.XYXYXY_pairs = torch.tensor(np.stack(task_img_dict[task_id]))
            self.XYXYXY_pairs = self.XYXYXY_pairs.unsqueeze(1) # [num_pairs, 1, 30, 180]

    def __len__(self):
        return len(self.XYXYXY_pairs)

    def __getitem__(self, idx):
        # print(f'self.XYXYXY[idx].shape: {self.XYXYXY[idx].shape}')
        XYXYX_ = self.XYXYXY_pairs[idx].clone()
        XYXYX_[:, :30, 150:] = torch.tensor(10.0)
        XYXYXY = self.XYXYXY_pairs[idx].clone()
        XYXYX_ = XYXYX_ / 10.0
        return XYXYX_, XYXYXY



# %%
train_dataset = ARCAGIDataset(training_challenges_json, training_solutions_json, task_id='007bbfb7')
eval_dataset =  ARCAGIDataset(evaluation_challenges_json, evaluation_solutions_json,)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
# %%
import torch.optim as optim 
from vit import ViTModel

# 훈련 예제
# 모델, 손실 함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTModel(num_classes=11).to(device)  # 클래스 수 11 (0~10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# %%
train_task_img_dict['007bbfb7'][0]
Y = train_task_img_dict['007bbfb7'][0]
X = Y.copy()
X[0:30, 150:] = 10

x = torch.tensor(X).float() / 10.0
x = x.unsqueeze(0).unsqueeze(0)
y = torch.tensor(Y)
y = y.unsqueeze(0).unsqueeze(0)

print(x.shape)
print(y.shape)

# %%
# single datapoint
# 훈련 루프 (예: 10 에포크)
# num_epochs = 1000
# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     x = x.cuda()
#     y = y.cuda()
#     # Forward: pred_y = model(x) -> (1, 11, 32, 192)
#     pred_y = model(x)
#     # print(pred_y.shape)
#     # Cross Entropy Loss: input (batch, classes, h, w), target (batch, h, w)
#     loss = criterion(pred_y, y.squeeze(1).long())

#     # Backward
#     loss.backward()
#     optimizer.step()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# %%
# 훈련 루프 (예: 10 에포크)

def count_equal_batches(a, b):
    # (32, 1, 30, 180)를 (32, -1)로 reshape하여 비교
    a_flat = a.view(a.shape[0], -1)  # (32, 5400)
    b_flat = b.view(b.shape[0], -1)  # (32, 5400)
    
    # 각 배치에서 모든 요소가 같은지 확인
    batch_equal = torch.all(a_flat == b_flat, dim=1)
    return torch.sum(batch_equal).item()

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    avg_train_loss = 0
    total_train_num_pairs = 0
    total_train_num_trues = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        optimizer.zero_grad()
        # Forward: pred_y = model(x) -> (1, 11, 32, 192)
        pred_y = model(batch_x)
        # print(pred_y.shape)
        # Cross Entropy Loss: input (batch, classes, h, w), target (batch, h, w)
        loss = criterion(pred_y, batch_y.squeeze(1).long())
        avg_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred_xyxyxy = torch.argmax(pred_y, dim=1)
        true_xyxyxy = batch_y.squeeze()
        num_train_trues = count_equal_batches(pred_xyxyxy, true_xyxyxy)
        total_train_num_pairs += len(batch_x)
        total_train_num_trues += num_train_trues
    
    model.eval()
    avg_eval_loss = 0
    total_eval_num_pairs = 0
    total_eval_num_trues = 0
    for batch_x, batch_y in eval_loader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        # Forward: pred_y = model(x) -> (1, 11, 32, 192)
        pred_y = model(batch_x)
        # print(pred_y.shape)
        # Cross Entropy Loss: input (batch, classes, h, w), target (batch, h, w)
        loss = criterion(pred_y, batch_y.squeeze(1).long())
        avg_eval_loss += loss.item()
        
        pred_xyxyxy = torch.argmax(pred_y, dim=1)
        true_xyxyxy = batch_y.squeeze()
        num_eval_trues = count_equal_batches(pred_xyxyxy, true_xyxyxy)
        total_eval_num_pairs += len(batch_x)
        total_eval_num_trues += num_eval_trues
        
    print(
        f"Epoch [{epoch+1}/{num_epochs}], " + 
        f"Train Loss: {avg_train_loss/len(train_loader):.4f}, " + 
        f"Train Success Rate: {total_train_num_trues/total_train_num_pairs}, " +
        f"Eval Loss: {avg_eval_loss/len(train_loader):.4f}, " + 
        f"Eval Success Rate: {total_eval_num_trues/total_eval_num_pairs}"
        )


# %%
for batch_x, batch_y in train_loader:
    break
# %%
x = batch_x[0].unsqueeze(0)
print(x.shape)
# %%
model.eval()
pred_y = model(x.cuda())
predicted_classes = torch.argmax(pred_y, dim=1)
# %%
predicted_classes.shape
# %%
plt.imshow((x[:, :, :, :180]*10).int().squeeze().squeeze().detach().cpu().numpy(), cmap=cmap, norm=norm,)
# %%
plt.imshow((predicted_classes[:, :, :180]).squeeze().detach().cpu().numpy(), cmap=cmap, norm=norm)

# %%
plt.imshow((y[:, :, :, :180]).squeeze().squeeze().detach().cpu().numpy(), cmap=cmap, norm=norm)


# %%

