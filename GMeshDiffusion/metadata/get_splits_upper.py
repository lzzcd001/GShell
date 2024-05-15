import os
import random

random.seed(42)

split_ratio = 0.9
data_root = 'PLACEHOLDER'
grid_root = os.path.join(data_root, 'grid')
occgrid_root = os.path.join(data_root, 'occgrid')
data_path_list = sorted([os.path.join(data_root, fpath) for fpath in os.listdir(data_root)])

random.shuffle(data_path_list)

n_train = int(len(data_path_list) * split_ratio)
train_list = data_path_list[:n_train]
test_list = data_path_list[n_train:]

with open('upper_res64_grid_train.txt', 'w') as f:
    f.write('\n'.join(train_list))

with open('upper_res64_grid_test.txt', 'w') as f:
    f.write('\n'.join(test_list))


occgrid_train_list = [os.path.join(occgrid_root, x.split('/')[-1]) for x in train_list]
occgrid_test_list = [os.path.join(occgrid_root, x.split('/')[-1]) for x in test_list]

with open('upper_res64_occgrid_train.txt', 'w') as f:
    f.write('\n'.join(occgrid_train_list))

with open('upper_res64_occgrid_test.txt', 'w') as f:
    f.write('\n'.join(occgrid_test_list))

