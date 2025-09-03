# %%
from arc_agi_grid_env_coord import action_converter

dict_action = action_converter(0)
print(dict_action)
dict_action = action_converter(899)
print(dict_action)
dict_action = action_converter(8100)
print(dict_action)
dict_action = action_converter(8999)
print(dict_action)
dict_action = action_converter(9000)
print(dict_action)
dict_action = action_converter(9899)
print(dict_action)
dict_action = action_converter(9900)
print(dict_action)
# %%
import torch 
action = torch.tensor(899)
dict_action = action_converter(action.cpu().item())
print(dict_action)
