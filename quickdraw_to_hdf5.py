import os.path
import pickle
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")

_project_folder_ = os.path.realpath(os.path.abspath('..'))
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)
from data.sketch_util import SketchUtil

# Arguments
quickdraw_root = '/home/QuickDraw/sketchrnn/'
output_root = '/home/QuickDraw/Category345/'
max_length_stroke = 0

if not os.path.exists(output_root):
    os.makedirs(output_root)

def get_categories():
    res = [npz_file.stem for npz_file in list(Path(quickdraw_root).glob('*.npz'))]
    return sorted(res, key=lambda s: s.lower())

def get_30_categories():
    res = ['airplane', 'apple', 'baseball bat', 'bed', 'bicycle', 'book', 'bread', 'broom', 'camera',
                            'car', 'cell phone', 'chair', 'clock', 'cloud', 'eye', 'eyeglasses', 'face', 'flower',
                            'headphones', 'hot dog', 'laptop', 'pants', 'shorts', 'smiley face', 'snake', 'spider',
                            'star', 'sun', 'table', 'tree']

    return sorted(res, key=lambda s: s.lower())

def get_9_categories():
    res = ['cat', 'chair', 'face', 'firetruck', 'mosquito', 'owl', 'pig', 'purse', 'shoe']

    return sorted(res, key=lambda s: s.lower())


def load_npz(file_path):
    npz = np.load(file_path, encoding='latin1', allow_pickle=True)
    return npz['train'], npz['valid'], npz['test']


def cvrt_points3(points3_array):
    points3 = np.array(points3_array, dtype=np.int32)
    points3[:, 0:2] = np.cumsum(points3[:, 0:2], axis=0)
    return points3


def cvrt_category_to_points3(points3_arrays, hdf5_group=None):
    max_num_points = 0
    res = []
    for pts3_arr in points3_arrays:
        # if len(pts3_arr) < 3:
        #     len_count += 1
        #     continue
        pts3 = np.array(cvrt_points3(pts3_arr), np.float32)
        pts3_norm = SketchUtil.normalization(pts3[:, 0:2])
        if pts3_norm is None:
            continue
        pts3[:, 0:2] = pts3_norm

        npts3 = len(pts3)
        if npts3 > max_num_points:
            max_num_points = npts3

        if hdf5_group is not None:
            hdf5_group.create_dataset(str(len(res)), data=npts3)

        res.append(npts3)
    return res, max_num_points

category_names = get_30_categories()
# category_names = get_9_categories()

print('[*] Number of categories = {}'.format(len(category_names)))
print('[*] ------')
print(category_names)
print('[*] ------')

hdf5_names = ['train', 'valid', 'test']
mode_indices = [list() for hn in hdf5_names]
hdf5_files = [h5py.File(os.path.join(output_root, 'quickdraw_{}.hdf5'.format(hn)), 'w', libver='latest') for hn in
              hdf5_names]
hdf5_groups = [h5.create_group('/sketch') for h5 in hdf5_files]

max_num_points = 0
for cid, category_name in enumerate(category_names):
    print('[*] Processing {}th category: {}'.format(cid + 1, category_name))

    train_valid_test = load_npz(os.path.join(quickdraw_root, category_name + '.npz'))

    for mid, mode in enumerate(hdf5_names):
        hdf5_category_group = hdf5_groups[mid].create_group(str(cid))
        pts3_arrays, npts3 = cvrt_category_to_points3(train_valid_test[mid], hdf5_category_group)
        nsketches = len(pts3_arrays)
        if mode == 'test':
            if nsketches != 2500:
                print('warning!!!!!!!!!')
                print(nsketches)

        if npts3 > max_num_points:
            max_num_points = npts3

        hdf5_category_group.attrs['num_sketches'] = nsketches
        mode_indices[mid].extend(list(zip([cid] * nsketches, range(nsketches))))

for gid, gp in enumerate(hdf5_groups):
    gp.attrs['num_categories'] = len(category_names)
    gp.attrs['max_points'] = max_num_points

for hf in hdf5_files:
    hf.flush()
    hf.close()

pkl_save = {'categories': category_names, 'indices': mode_indices}
with open(os.path.join(output_root, 'categories.pkl'), 'wb') as fh:
    pickle.dump(pkl_save, fh, pickle.HIGHEST_PROTOCOL)

print('max_num_points = {}'.format(max_num_points))
print('All done.')
print(f'max_stroke_points  {max_length_stroke}')
