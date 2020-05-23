from pathlib import Path

file_name_env = 'D:\D_Downloads\Tennis_Windows_x86_64\Tennis.exe'

model_path = Path('./models/')
output_path = Path('./output/')
file_path_model = model_path / 'model.npy'
file_path_ref_model = model_path / 'model_ref.npy'
file_path_img_score = output_path / 'score.png'
file_path_ref_img_score = output_path / 'score_ref.png'
file_path_img_loss = output_path / 'loss.png'
file_path_img_actions = output_path / 'actions.png'
file_path_img_noise = output_path / 'noise.png'

# general params
random_seed = 0  # 0xABCD
rolling_mean_N = 100
num_episodes_test = 10
num_agents = 2
state_size = 24
action_size = 2
verbose = False
high_score = 0.5

# algo params
num_episodes = 1500
max_action = 1.

# agent params
memory_size = 200000  # 100000
gamma = 0.99
batch_size = 128
tau = 0.06  # 0.001
policy_freq = 1

# model params
model_learning_rate = 0.001  # 0.0001
num_fc_1 = 256
num_fc_2 = 128
with_bn = True

# noise params
ou_stddev = 0.2
ou_theta = 0.13  # 0.15
ou_dt = 1  # 0.01
eps_start = 6
eps_end = 0


def myprint(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)
