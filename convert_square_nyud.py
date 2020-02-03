import errno
import re
import json
import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from data_io.utils import scaleit3

def get_subdir(path):
    paths = os.listdir(path)
    return [
        {'relative': p, 'absolute': os.path.join(path, p)}
        for p in paths if os.path.isdir(os.path.join(path, p))
    ]



def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dry-run', action='store_true')

    ARGS = parser.parse_args()

    assert os.path.exists(ARGS.input_dir), 'input_dir does not exist'
    assert os.path.exists(ARGS.output_dir), 'output_dir does not exist'

    # go through the splits
    splits = get_subdir(ARGS.input_dir)
    for split in splits:
        modalities = get_subdir(split['absolute'])

        # go through the modalities
        for modality in modalities:
            classes = get_subdir(modality['absolute'])

            # go through the class-labels
            for class_label in classes:
                img_files = os.listdir(class_label['absolute'])

                # create the full input paths and create the corresponding
                # output paths
                full_input_paths = [os.path.join(
                    class_label['absolute'], im_name
                ) for im_name in img_files]
                full_output_paths = [
                    os.path.join(
                        ARGS.output_dir,
                        split['relative'],
                        modality['relative'],
                        class_label['relative'],
                        im_name
                    )
                    for im_name in img_files
                ]

                # go through the outputs and make sure that the directories
                # all exist
                for output_path in full_output_paths:
                    _path, _fname = os.path.split(output_path)
                    if not os.path.exists(_path):
                        print(f'Output-Path: {_path} does not exist')
                        if ARGS.dry_run:
                            print('...would create directory now')
                        else:
                            print('...creating directory now')
                            mkdir_p(_path)

                # go through all the input files
                for i, input_file in enumerate(full_input_paths):
                    print('{} -> {}'.format(
                        input_file,
                        full_output_paths[i]
                    ))

                    pil_im = Image.open(input_file)
                    np_im = scaleit3(np.asarray(pil_im), (128, 128, 3))

                    pil_resized_im = Image.fromarray(np_im)

                    if ARGS.dry_run:
                        _imshape_old = pil_im.size
                        _imshape_new = pil_resized_im.size
                        print(f'... {input_file} would be saved as resized img'
                              f' from {_imshape_old} to {_imshape_new}')
                    else:
                        pil_resized_im.save(full_output_paths[i])

    # compute the new averages
    if ARGS.dry_run:
        print('Do not compute the averages per modality')
    else:
        for modality in ['image', 'depth']:
            modality_data = []

            # get the train class for the modalities
            modality_input_path = os.path.join(
                ARGS.output_dir, 'train', modality)
            classes = get_subdir(modality_input_path)

            # go through the classes
            for class_label in classes:

                im_paths = os.listdir(class_label['absolute'])

                for img in im_paths:
                    _im_path = os.path.join(class_label['absolute'], img)
                    im_pil = Image.open(_im_path)
                    modality_data.append(np.asarray(im_pil))
            # collected data
            M = np.asarray(modality_data)
            M_flat = np.reshape(M, (-1, 3))

            mu_mod = np.mean(M_flat, axis=0).tolist()
            std_mod = np.std(M_flat, axis=0).tolist()

            print(f'Modality: {modality}')
            print('\tmean: {}'.format(
                '/255., '.join(list(map(str, mu_mod))) + '/255.'
            ))
            print('\tstd: {}'.format(
                '/255., '.join(list(map(str, std_mod))) + '/255.'
            ))
