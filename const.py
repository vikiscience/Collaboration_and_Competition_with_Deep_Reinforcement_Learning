from pathlib import Path

file_name_env = 'D:\D_Downloads\Tennis_Windows_x86_64\Tennis.exe'

model_path = Path('./models/')
output_path = Path('./output/')
file_path_model = model_path / 'model.npy'
file_path_ref_model = model_path / 'model_ref.npy'
file_path_img_score = output_path / 'score.png'
file_path_ref_img_score = output_path / 'score_ref.png'

# general params
random_seed = 0xABCD
rolling_mean_N = 100
num_agents = 2
state_size = 24
action_size = 2
max_action = 1.
episode_length = 300
verbose = False
high_score = 0.5

# algo params
num_episodes = 250

# agent params
memory_size = 40000
gamma = 0.95
batch_size = 32
expl_noise = 0.3

# model params
model_learning_rate = 0.001
num_fc_1 = 64
num_fc_2 = 32


def myprint(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)
