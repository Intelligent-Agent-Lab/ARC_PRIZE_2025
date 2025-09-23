import json
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

# --- 내부 객체 탐지 ---

def _get_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Calculates the bounding box of a boolean mask."""
    rs, cs = np.where(mask)
    if rs.size == 0:
        return (0, 0, -1, -1)  # Return invalid box if mask is empty
    return (int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max()))

def _find_components_by_color(grid: np.ndarray, color: int) -> List[np.ndarray]:
    """Finds all connected components for a single color, returning a list of boolean masks."""
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    masks = []
    # Use 4-way connectivity for neighbors
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for r in range(H):
        for c in range(W):
            if visited[r, c] or grid[r, c] != color:
                continue
            
            # Start BFS to find a new component
            q = [(r, c)]
            visited[r, c] = True
            component_points = [(r, c)]
            while q:
                rr, cc = q.pop(0) # BFS queue
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and grid[nr, nc] == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
                        component_points.append((nr, nc))
            
            # Create a boolean mask for the found component
            mask = np.zeros((H, W), dtype=bool)
            rows = [p[0] for p in component_points]
            cols = [p[1] for p in component_points]
            mask[rows, cols] = True
            masks.append(mask)
    return masks


@dataclass(frozen=True)
class Shape:
    height: int
    width: int

    def __repr__(self):
        return f"({self.height}, {self.width})"

def find_objects(grid):
    """
    Finds all connected objects in a grid, sorted by size in descending order.
    """
    grid = np.array(grid)
    objects = []
    unique_colors = [int(c) for c in np.unique(grid) if c != 0]

    for color in unique_colors:
        masks = _find_components_by_color(grid, color)
        for mask in masks:
            rs, cs = np.where(mask)
            if rs.size > 0:
                objects.append({
                    "color": color,
                    "points": [],
                    "size": len(rs),
                    "bbox": _get_bounding_box(mask)
                })
    return sorted(objects, key=lambda x: x["size"], reverse=True)

# --- 그리드 변환 ---

def _get_inner_objects(t):
    """Returns a list of objects that do not touch the grid border."""
    objects = find_objects(t['input'])
    if not objects: return []
    grid_h, grid_w = t['input_shape'].height, t['input_shape'].width
    return [o for o in objects if o['bbox'][0] > 0 and o['bbox'][1] > 0 and o['bbox'][2] < grid_h - 1 and o['bbox'][3] < grid_w - 1]

# --- Shape Transformations (Leak-Free) ---
def id_transform(t): return Shape(height=t['input_shape'].height, width=t['input_shape'].width)
def transpose_transform(t): return Shape(height=t['input_shape'].width, width=t['input_shape'].height)

def crop_to_all_inner_objects_bbox(t):
    inner_objects = _get_inner_objects(t)
    if not inner_objects: return None
    min_r = min(o['bbox'][0] for o in inner_objects)
    min_c = min(o['bbox'][1] for o in inner_objects)
    max_r = max(o['bbox'][2] for o in inner_objects)
    max_c = max(o['bbox'][3] for o in inner_objects)
    return Shape(height=max_r - min_r + 1, width=max_c - min_c + 1)

def crop_to_largest_non_boundary_object_bbox(t):
    objects = find_objects(t['input'])
    if not objects: return None
    grid_h, grid_w = t['input_shape'].height, t['input_shape'].width
    non_boundary_objects = [o for o in objects if not (o['bbox'][0] == 0 or o['bbox'][1] == 0 or o['bbox'][2] == grid_h - 1 or o['bbox'][3] == grid_w - 1)]
    if not non_boundary_objects: return None
    lrg_obj = max(non_boundary_objects, key=lambda x: x['size'])
    min_r, min_c, max_r, max_c = lrg_obj['bbox']
    return Shape(height=max_r - min_r + 1, width=max_c - min_c + 1)

def crop_to_object_bbox(t):
    objects = find_objects(t['input'])
    if not objects: return None
    min_r, min_c, max_r, max_c = objects[0]['bbox']
    return Shape(height=max_r - min_r + 1, width=max_c - min_c + 1)

def _create_shape_from_object_count_rule(shape_template):
    """Factory to create shape prediction rules based on object count."""
    def rule(t):
        n = len(find_objects(t['input']))
        if n == 0:
            return None
        height = n if shape_template[0] == 'n' else shape_template[0]
        width = n if shape_template[1] == 'n' else shape_template[1]
        return Shape(height, width)
    return rule

shape_from_object_count_nxn = _create_shape_from_object_count_rule(('n', 'n'))
shape_from_object_count_1xn = _create_shape_from_object_count_rule((1, 'n'))
shape_from_object_count_nx1 = _create_shape_from_object_count_rule(('n', 1))

# --- 색상 규칙 ---
def color_id_transform(t): return set(np.array(t['input']).flatten())

def color_of_inner_objects(t):
    inner_objects = _get_inner_objects(t)
    if not inner_objects: return None
    inner_colors = {o['color'] for o in inner_objects}
    inner_colors.add(0)
    return inner_colors

def color_of_largest_object(t):
    objects = find_objects(t['input'])
    return {objects[0]['color'], 0} if objects else {0}

def color_palette_subtraction(t):
    input_colors = Counter(np.array(t['input']).flatten())
    if 0 in input_colors: del input_colors[0]
    if not input_colors: return set(np.array(t['input']).flatten())
    most_common_color = input_colors.most_common(1)[0][0]
    return {c for c in np.array(t['input']).flatten() if c != most_common_color}

# --- Prediction Logic ---
# 간단한 것 부터 복잡한 것 순으로 대조
def _prepare_train_tasks(task):
    train_tasks = []
    for ex in task.get('train', []):
        if 'input' not in ex or 'output' not in ex or not ex['input'] or not ex['input'][0] or not ex['output'] or not ex['output'][0]:
            continue
        train_tasks.append({
            "input": ex['input'], "output": ex['output'],
            "input_shape": Shape(len(ex['input']), len(ex['input'][0])),
            "output_shape": Shape(len(ex['output']), len(ex['output'][0]))
        })
    return train_tasks

def predict_shape(train_tasks):
    if not train_tasks: return "fallback_id", id_transform

    # --- Simple Rules ---
    simple_rules = [
        ("id_transform", id_transform, False),
        ("transpose_transform", transpose_transform, False),
        ("crop_to_all_inner_objects_bbox", crop_to_all_inner_objects_bbox, True),
        ("crop_to_largest_non_boundary_object_bbox", crop_to_largest_non_boundary_object_bbox, True),
        ("crop_to_object_bbox", crop_to_object_bbox, True),
        ("shape_from_object_count_nxn", shape_from_object_count_nxn, True),
        ("shape_from_object_count_1xn", shape_from_object_count_1xn, True),
        ("shape_from_object_count_nx1", shape_from_object_count_nx1, True),
    ]
    for name, rule_func, check_none in simple_rules:
        try:
            if check_none:
                if all(rule_func(t) is not None and rule_func(t) == t['output_shape'] for t in train_tasks):
                    return name, rule_func
            else:
                if all(rule_func(t) == t['output_shape'] for t in train_tasks):
                    return name, rule_func
        except (AttributeError, TypeError, IndexError):
            continue

    # --- Complex Rules (Delta-based) ---
    deltas = {(t['output_shape'].height - t['input_shape'].height, t['output_shape'].width - t['input_shape'].width) for t in train_tasks}
    if len(deltas) == 1:
        dh, dw = deltas.pop()
        predictor = lambda t, h=dh, w=dw: Shape(t['input_shape'].height + h, t['input_shape'].width + w)
        return ("crop_transform" if dh < 0 or dw < 0 else "pad_transform"), predictor

    # --- Tiling Rule ---
    factors = set()
    is_tileable = True
    for t in train_tasks:
        if t['input_shape'].height > 0 and t['input_shape'].width > 0 and \
           t['output_shape'].height % t['input_shape'].height == 0 and \
           t['output_shape'].width % t['input_shape'].width == 0:
            factors.add((t['output_shape'].height // t['input_shape'].height, t['output_shape'].width // t['input_shape'].width))
        else:
            is_tileable = False
            break
    if is_tileable and len(factors) == 1:
        fh, fw = factors.pop()
        if fh > 0 and fw > 0:
            return "tile_transform", lambda t, h=fh, w=fw: Shape(t['input_shape'].height * h, t['input_shape'].width * w)

    # --- Fallbacks ---
    output_shapes = [t['output_shape'] for t in train_tasks]
    most_common_shape = Counter(output_shapes).most_common(1)[0][0]
    if all(t['output_shape'] == most_common_shape for t in train_tasks):
        return "constant_shape", lambda t, shape=most_common_shape: shape

    first_output_shape = train_tasks[0]['output_shape']
    return "fallback_first_train_shape", lambda t, shape=first_output_shape: shape

def predict_color_palette(train_tasks):
    if not train_tasks: return "color_union_transform", lambda t: set()

    # --- Simple Rules ---
    rules = [
        ("color_id_transform", color_id_transform, False),
        ("color_of_inner_objects", color_of_inner_objects, True),
        ("color_of_largest_object", color_of_largest_object, False),
        ("color_palette_subtraction", color_palette_subtraction, False),
    ]
    for name, rule_func, check_none in rules:
        try:
            if all((res := rule_func(t)) is not None and res == set(np.array(t['output']).flatten()) for t in train_tasks) if check_none \
            else all(rule_func(t) == set(np.array(t['output']).flatten()) for t in train_tasks):
                return name, rule_func
        except (AttributeError, TypeError, IndexError):
            continue

    # --- Constant Color Rule ---
    first_output_colors = set(np.array(train_tasks[0]['output']).flatten())
    if all(set(np.array(t['output']).flatten()) == first_output_colors for t in train_tasks[1:]):
        return "color_constant_transform", lambda t, colors=first_output_colors: colors

    # --- Fallback ---
    all_output_colors = set.union(*[set(np.array(t['output']).flatten()) for t in train_tasks])
    return "color_union_transform", lambda t, colors=all_output_colors: colors

def predict_candidates_from_task_id(task_id, training_challenges):
    task = training_challenges.get(task_id)
    if not task: return ([10, 10], list(range(10)))

    train_tasks = _prepare_train_tasks(task)
    if not train_tasks: return ([10, 10], list(range(10)))

    _, shape_rule_func = predict_shape(train_tasks)
    _, color_rule_func = predict_color_palette(train_tasks)

    test_inputs = task.get('test', [])
    test_input_grid = test_inputs[0]['input'] if test_inputs and test_inputs[0].get('input') else train_tasks[0]['input']
    
    if not test_input_grid or not test_input_grid[0]: return ([10, 10], list(range(10)))

    leak_free_test_instance = {"input": test_input_grid, "input_shape": Shape(len(test_input_grid), len(test_input_grid[0]))}
    
    predicted_shape = shape_rule_func(leak_free_test_instance)
    predicted_palette = color_rule_func(leak_free_test_instance)

    # Format results with fallbacks
    if predicted_shape is None:
        predicted_shape = train_tasks[0]['output_shape']
    size_candidate = [predicted_shape.height, predicted_shape.width]

    color_candidate = []
    if predicted_palette is not None:
        color_candidate = sorted([c for c in predicted_palette if 0 <= c <= 9])
    
    if not color_candidate:
        color_candidate = sorted({int(c) for c in np.array(train_tasks[0]['output']).flatten() if 0 <= c <= 9})

    if not size_candidate or len(size_candidate) != 2: size_candidate = [10, 10]
    if not color_candidate: color_candidate = list(range(10))

    return size_candidate, color_candidate

# --- Analysis & Main Execution (Commented Out) ---
#
# def _print_task_analysis(task_id, train_tasks, results):
#     print(f"--- 평가 중: 태스크 ID: {task_id} ---")
#     print("[Train Data]")
#     for i, t in enumerate(train_tasks):
#         in_colors = sorted(list({int(c) for c in np.array(t['input']).flatten()}))
#         out_colors = sorted(list({int(c) for c in np.array(t['output']).flatten()}))
#         print(f"- 예제 {i+1}: Shape: {t['input_shape']} -> {t['output_shape']} | Colors: {in_colors} -> {out_colors}")
#     
#     print("\n[Prediction Result]")
#     shape_status = "성공!" if results['shape_correct'] else "실패!"
#     color_status = "성공!" if results['color_correct'] else "실패!"
#     
#     print(f"Shape: {shape_status} | 규칙: {results['shape_rule_name']} | 예측: {results['predicted_shape']} (정답: {results['actual_shape']})")
#     print(f"Color: {color_status} | 규칙: {results['color_rule_name']} | 예측: {sorted(list(results['predicted_palette']))} (정답: {sorted(list(results['actual_palette']))})")
#     print("-" * 40)
#
# def analyze_dataset(challenge_file, solution_file):
#     with open(challenge_file, 'r') as f: challenges = json.load(f)
#     with open(solution_file, 'r') as f: solutions = json.load(f)
#
#     stats = {'total': 0, 'shape_correct': 0, 'color_correct': 0, 'total_correct': 0}
#
#     for task_id, task in challenges.items():
#         if task_id not in solutions: continue
#         
#         train_tasks = _prepare_train_tasks(task)
#         if not train_tasks: continue
#
#         stats['total'] += 1
#         shape_rule_name, shape_rule_func = predict_shape(train_tasks)
#         color_rule_name, color_rule_func = predict_color_palette(train_tasks)
#
#         test_input = task['test'][0]['input']
#         leak_free_test_instance = {"input": test_input, "input_shape": Shape(len(test_input), len(test_input[0]))}
#
#         predicted_shape = shape_rule_func(leak_free_test_instance)
#         predicted_palette = color_rule_func(leak_free_test_instance)
#         
#         actual_output = solutions[task_id][0]
#         actual_shape = Shape(len(actual_output), len(actual_output[0]))
#         actual_palette = set(np.array(actual_output).flatten())
#
#         shape_correct = (predicted_shape == actual_shape)
#         color_correct = (predicted_palette == actual_palette)
#         if shape_correct: stats['shape_correct'] += 1
#         if color_correct: stats['color_correct'] += 1
#         if shape_correct and color_correct: stats['total_correct'] += 1
#
#         _print_task_analysis(task_id, train_tasks, {
#             'shape_correct': shape_correct, 'color_correct': color_correct,
#             'shape_rule_name': shape_rule_name, 'color_rule_name': color_rule_name,
#             'predicted_shape': predicted_shape, 'actual_shape': actual_shape,
#             'predicted_palette': predicted_palette, 'actual_palette': actual_palette
#         })
#
#     if stats['total'] > 0:
#         print(f"\n--- 최종 결과: {Path(challenge_file).stem} ---")
#         print(f"Shape 예측 성공률: {stats['shape_correct']/stats['total']:.2%} ({stats['shape_correct']}/{stats['total']})")
#         print(f"Color 예측 성공률: {stats['color_correct']/stats['total']:.2%} ({stats['color_correct']}/{stats['total']})")
#         print(f"Total 예측 성공률: {stats['total_correct']/stats['total']:.2%} ({stats['total_correct']}/{stats['total']})")
#
# if __name__ == "__main__":
#     dataset_path = Path(__file__).parent.parent / 'datasets'
#     
#     print("="*20 + " 훈련 데이터셋 분석 " + "="*20)
#     analyze_dataset(
#         dataset_path / 'arc-agi_training_challenges.json',
#         dataset_path / 'arc-agi_training_solutions.json'
#     )