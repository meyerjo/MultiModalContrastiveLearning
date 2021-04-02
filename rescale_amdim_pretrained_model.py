import os
from argparse import ArgumentParser

import torch

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--n_classes', type=int, default=51)
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
        if key in relevant_tensors:
            old_item_shape = item.shape
            print(key, type(model['model'][key]), item.shape)
            if 'weight' in key:
                w = torch.empty(args.n_classes, item.shape[1])
                torch.nn.init.xavier_uniform_(w)
                item = w
                model['model'][key] = w
            elif 'bias' in key:
                w = torch.zeros([args.n_classes])
                # torch.nn.init.xavier_uniform_(w)
                item = w
                model['model'][key] = w
            else:
                raise BaseException('Neither weight nor bias in the key name')
            
            # model['model'][key] = item[:19, ...] # this was removed on 2020-02-27
            print(key, item, old_item_shape, item.shape)

    if 'hyperparams' in model:
        model['hyperparams']['n_classes'] = 19

    if args.output_path is not None:
        torch.save(model, args.output_path)
    else:
        print('Dry run as no output_path is provided. Not saving something')
