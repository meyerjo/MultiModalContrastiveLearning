import functools
import operator
import re
import os
import numpy as np
from queue import Queue

from PIL import Image

DATASET_FOLDER = '/home/meyerjo/dataset/sun_rgbd/'

def find_paths_iter(path):
    queue = Queue()
    results = []
    queue.put(path)
    while not queue.empty():
        elm = queue.get()
        elements = os.listdir(elm)
        if 'scene.txt' in elements:
            results += [elm]
            continue
        for _e in elements:
            _full_path = os.path.join(elm, _e)
            if os.path.isdir(_full_path):
                queue.put(_full_path)
    return results


if __name__ == '__main__':
    from datetime import datetime

    start = datetime.now()
    paths = find_paths_iter(os.path.join(DATASET_FOLDER, 'SUNRGBD'))
    print('Iter: {} {}'.format(datetime.now() - start, len(paths)))

    # write down statistics
    classes = {}
    for path in paths:
        _f_scenetype = os.path.join(path, 'scene.txt')
        _f_content = open(_f_scenetype, 'r').read()

        if _f_content not in classes.keys():
            classes[_f_content] = 1
        else:
            classes[_f_content] += 1

    classes = sorted([(key, item) for key, item in classes.items()], key=lambda x: x[1], reverse=True)
    for key, item in classes:
        print('{:20s} {}'.format(key, item))

    # train / test
    fname_meta_train = os.path.join(DATASET_FOLDER, 'sunrgbd-meta-data', 'sunrgbd_training_images.txt')
    fname_meta_test = os.path.join(DATASET_FOLDER, 'sunrgbd-meta-data', 'sunrgbd_testing_images.txt')

    meta_train = open(fname_meta_train).readlines()
    meta_train = ['/'.join(e.split('/')[:-2]) for e in meta_train]

    meta_test = open(fname_meta_test).readlines()
    meta_test = ['/'.join(e.split('/')[:-2]) for e in meta_test]

    rgb = []
    depth = []
    from skimage.transform import resize
    for mt in meta_train:
        mt = mt[:-1] if mt.endswith('/') else mt
        _tmp_f = os.path.join(DATASET_FOLDER, mt)
        if _tmp_f not in paths:
            print(mt)
        #     continue
        # _rgb_folder = os.path.join(_tmp_f, 'image')
        # _depth_folder = os.path.join(_tmp_f, 'depth')
        # # get the files
        # rgb_files = os.listdir(_rgb_folder)
        # depth_files = os.listdir(_depth_folder)
        #
        # rgb_files = [os.path.join(_rgb_folder, e) for e in rgb_files if os.path.isfile(os.path.join(_rgb_folder, e))]
        # depth_files = [os.path.join(_depth_folder, e) for e in depth_files if os.path.isfile(os.path.join(_depth_folder, e))]
        #
        # assert(len(rgb_files) == 1)
        # assert(len(depth_files) == 1)
        #
        # rgb_file = np.array(Image.open(rgb_files[0]))
        # rgb_file = resize(rgb_file, output_shape=(128, 128))
        # # rgb += [rgb_file]
        #
        #
        # depth_file = np.array(Image.open(depth_files[0]))
        # depth += [depth_file]



    for mt in meta_test:
        mt = mt[:-1] if mt.endswith('/') else mt
        _tmp_f = os.path.join(DATASET_FOLDER, mt)
        if _tmp_f not in paths:
            print(mt)
            continue
        _rgb_folder = os.path.join(_tmp_f, 'image')
        _depth_folder = os.path.join(_tmp_f, 'depth')
        # get the files
        rgb_files = os.listdir(_rgb_folder)
        depth_files = os.listdir(_depth_folder)

        rgb_files = [os.path.join(_rgb_folder, e) for e in rgb_files if
                     os.path.isfile(os.path.join(_rgb_folder, e))]
        depth_files = [os.path.join(_depth_folder, e) for e in depth_files if
                       os.path.isfile(os.path.join(_depth_folder, e))]

        assert (len(rgb_files) == 1)
        assert (len(depth_files) == 1)

        rgb_file = np.array(Image.open(rgb_files[0]))
        # rgb_file = resize(rgb_file, output_shape=(128, 128))
        rgb += [rgb_file]

        depth_file = np.array(Image.open(depth_files[0]))
        # depth += [depth_file]


    # concat_elements = np.concatenate([np.reshape(r, (-1, 3)) for r in rgb ], axis=0)
    # mean_rgb = np.mean(concat_elements, axis=0)
    # std_rgb = np.std(concat_elements, axis=0)
    concat_elements = np.concatenate([np.reshape(r, (-1, 3)).astype(np.uint8) for r in rgb], axis=0)
    # concat_elements = np.concatenate([np.reshape(r, (-1, 3)).astype(np.uint8) for r in depth], axis=0)
    mean_rgb = np.mean(concat_elements, axis=0)
    std_rgb = np.std(concat_elements, axis=0)

# rgb mean  [125.9179271  116.55520094 110.44606892]
# rgb std   [71.01960914 72.97451793 74.19717702]