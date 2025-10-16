import numpy as np 
import torch


def randomize_part_of_solution(target_grid, current_grid, active_shape, 
                                empty_val=11, fill_ratio_range=(0.2, 0.7)):
    """
    타겟 그리드의 활성 영역을 부분적으로 정답으로 초기화하는 함수
    
    Args:
        target_grid: 정답이 포함된 타겟 그리드 (numpy array)
        current_grid: 현재 그리드 (numpy array, in-place로 수정됨)
        active_shape: 활성 영역을 나타내는 마스크 (numpy array, 1인 부분이 활성 영역)
        empty_val: 빈 셀을 나타내는 값 (기본값: 11)
        fill_ratio_range: 채울 비율의 범위 (min, max) 튜플 (기본값: (0.2, 0.7))
    
    Returns:
        None (current_grid를 in-place로 수정)
    """
    if target_grid is None:
        return
    
    # Extract target values in active region (solution 영역은 column 90 이후)
    target_solution = target_grid[:, 90:][active_shape == 1]
    
    # Randomly fill some percentage of correct answers
    fill_ratio = np.random.uniform(fill_ratio_range[0], fill_ratio_range[1])
    total_cells = np.sum(active_shape)
    num_filled = int(total_cells * fill_ratio)
    
    # Create mask for which cells to fill
    fill_mask = np.zeros(total_cells, dtype=bool)
    filled_indices = np.random.choice(total_cells, num_filled, replace=False)
    fill_mask[filled_indices] = True
    
    # Start with empty values
    current_solution = np.full(total_cells, empty_val)
    
    # Fill selected positions with correct answers
    current_solution[fill_mask] = target_solution[fill_mask]
    
    # Apply to current grid
    current_grid[:, 90:][active_shape == 1] = current_solution
    

def convert_int_to_dict(action: int) -> dict:
    # 0 ~ 899: color 0
    # 900 ~ 1799: color 1
    # 1800 ~ 2699: color 2
    # 2700 ~ 3599: color 3
    # 3600 ~ 4499: color 4
    # 4500 ~ 5399: color 5
    # 5400 ~ 6299: color 6
    # 6300 ~ 7199: color 7
    # 7200 ~ 8099: color 8
    # 8100 ~ 8999: color 9
    # 9000 ~ 9899: color 10
    assert (action >= 0 and action < 9900)
    color = action // 900
    row = (action - 900*color) // 30
    col = (action - 900*color) % 30
    coordinate = (row, col)
    dict_action = {'color': color, 
                   'coordinate': coordinate}
    return dict_action
    

def vectorized_convert_int_to_dict(actions: torch.Tensor) -> dict:
    """벡터화된 함수 (텐서용)"""
    # 입력 검증
    assert torch.all((actions >= 0) & (actions < 9900)), "All actions must be in range [0, 9900)"
    
    # 벡터화된 계산
    # color = action // 900
    colors = torch.div(actions, 900, rounding_mode='floor')
    remainders = actions - 900 * colors
    
    # row = remainder // 30
    rows = torch.div(remainders, 30, rounding_mode='floor')
    cols = remainders % 30
    
    # 결과를 딕셔너리로 반환
    result = {
        'color': colors.numpy(),
        'coordinate': torch.stack([rows, cols], dim=-1).numpy()  # (N, 2) 형태
    }
    return result


def convert_dict_to_int(dict_action: dict,
                        ) -> int:
    """
    딕셔너리 액션을 정수 액션으로 변환하는 벡터화된 함수
    Args:
        dict_actions: {
            'color': numpy array or torch.Tensor of shape (,),
            'coordinate': numpy array or torch.Tensor of shape (2,)
        }
    Returns:
        np.ndarray: 정수 액션들 (N,)
    """
    color = dict_action['color']
    coordinate = dict_action['coordinate']
    int_action = color * 900 + coordinate[0]*30 + coordinate[1]
    return int_action


def vectorized_convert_dict_to_int(dict_actions: dict) -> np.ndarray:
    """
    딕셔너리 액션들을 정수 액션들로 변환하는 벡터화된 함수
    Args:
        dict_actions: {
            'color': numpy array or torch.Tensor of shape (N,),
            'coordinate': numpy array or torch.Tensor of shape (N, 2)
        }
    Returns:
        np.ndarray: 정수 액션들 (N,)
    """
    colors = dict_actions['color']
    coordinates = dict_actions['coordinate']
    
    # 벡터화된 계산: int_action = color * 900 + row * 30 + col
    # coordinates는 (N, 2) 형태이므로 coordinates[:, 0]이 row, coordinates[:, 1]이 col
    rows = coordinates[:, 0]
    cols = coordinates[:, 1]
    int_actions = colors * 900 + rows * 30 + cols
    return int_actions
