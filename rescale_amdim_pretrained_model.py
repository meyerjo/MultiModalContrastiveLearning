import os
from argparse import ArgumentParser

import torch

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print('Input Path: {} does not exist'.format(args.input_path))
        exit(1)


    model = torch.load(args.input_path)
    # model = torch.load('./amdim_ndf320_rkhs2560_rd10.pth')

    relevant_tensors = [
        'evaluator.block_glb_mlp.block_forward.6.weight',
        'evaluator.block_glb_mlp.block_forward.6.bias',
        'evaluator.block_glb_lin.block_forward.2.weight',
        'evaluator.block_glb_lin.block_forward.2.bias']


    for key, item in model['model'].items():
        print(key, item.shape)

    for key, item in model['model'].items():
        if key in relevant_tensors:
            old_item_shape = item.shape
            model['model'][key] = item[:19, ...]
            print(key, item, old_item_shape, item.shape)

    for key, item in model['model'].items():
        print(key, item.shape)

    model['hyperparams']['n_classes'] = 19

    if args.output_path is not None:
        torch.save(model, args.output_path)
    else:
        print('Dry run as no output_path is provided. Not saving something')
