import os
import torch
import random
import shutil
from pprint import pprint
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from neuralphys.utils.config import _C as C
from neuralphys.utils.logger import setup_logger, git_diff_config
from neuralphys.models import *
from neuralphys.models import rpin
from neuralphys.trainer import Trainer
from hyperopt import STATUS_OK


def arg_parse():
    # only the most general argument is passed here
    # task-specific parameters should be passed by config file
    parser = argparse.ArgumentParser(description='RPIN parameters')
    parser.add_argument('--cfg', required=False, help='path to config file', type=str)
    #parser.add_argument('--init', type=str, default='')
    parser.add_argument('--predictor-arch', type=str, default='rpin')
    parser.add_argument('--gpus', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--seed', type=int, default=0)
    args, unknown = parser.parse_known_args()
    print("Unknown arguments", unknown)
    return args


COLLIDE_CONFIG = {
        'binary_labels': ['is_colliding_dynamic'],
        'train_shift': [30, 1024, 1],
        'train_len': 32470,
        'test_shift': [30, 1024, 16],
        'test_len': 62,
        }

TOWER_CONFIG = {
        'binary_labels': ['is_stable'],
        'train_shift': [0, 1024, 1],
        'train_len': 63360,
        'test_shift': [2, 1024, 32],
        'test_len': 384,
        }

CONTAIN_CONFIG = {
        'binary_labels': ['object_data'],
        'train_shift': [30, 1024, 1],
        'train_len': 36355,
        'test_shift': [30, 1024, 16],
        'test_len': 28,
        }

CLOTH_CONFIG = {
        'binary_labels': ['object_category'],
        'train_shift': [0, 1024, 1],
        'train_len': 63360,
        'test_shift': [2, 1024, 64],
        'test_len': 192,
        }

ROLL_VS_SLIDE_CONFIG = {
        'binary_labels': ['is_rolling'],
        'train_shift': [32, 1024, 1],
        'train_len': 74572 // 4,
        'test_shift': [32, 1024, 64],
        'test_len': 320 // 4,
        }


def get_config(subset):
    if 'collide' in subset:
        return COLLIDE_CONFIG
    elif 'tower' in subset:
        return TOWER_CONFIG
    elif 'contain' in subset:
        return CONTAIN_CONFIG
    elif 'cloth' in subset:
        return CLOTH_CONFIG
    elif 'roll' in subset:
        return ROLL_VS_SLIDE_CONFIG
    elif 'slide' in subset:
        return ROLL_VS_SLIDE_CONFIG
    else:
        raise ValueError("Unkown config for subset: %s" % subset)


def get_data_paths(root, subsets):
    data_paths = [os.path.join(root, subset) for subset in subsets]
    return data_paths


def get_data_len(subsets):
    train_data_len = 0
    val_data_len = 0
    for subset in subsets:
        train_data_len += get_config(subset)['train_len']
        val_data_len += get_config(subset)['test_len']
    return train_data_len, val_data_len


def get_binary_labels(subsets):
    if len(subsets) > 1:
        return ['is_colliding_dynamic']
    else:
        return get_config(subsets[0])['binary_labels']


def get_shift_selector(subsets):
    if len(subsets) > 1:
        train_shift_selector = [0, 1024, 1]
        val_shift_selector = [0, 1024, 32]
    else:
        train_shift_selector = get_config(subsets[0])['train_shift']
        val_shift_selector = get_config(subsets[0])['test_shift']
    return train_shift_selector, val_shift_selector


def get_model_dir(run_name, seed, base_dir = '/mnt/fs4/mrowca/hyperopt/rpin'):
    return os.path.join(base_dir, run_name, str(seed), 'model')


def run(
        train_name = 'collide',
        datasets = ['collide2_new'],
        seed = 0,
        data_root = '/mnt/fs4/mrowca/neurips/images/rigid',
        base_dir = '/mnt/fs4/mrowca/hyperopt/rpin',
        write_feat = '',
        test_name = 'default',
        ):
    # the wrapper file contains:
    # 1. setup environment
    # 2. setup config
    # 3. setup logger
    # 4. setup model
    # 5. setup optimizer
    # 6. setup dataset

    # ---- setup environment
    args = arg_parse()

    # Overwrite args with passed args
    args.seed = seed if not write_feat else 0
    args.cfg = '/home/mrowca/workspace/RPIN/configs/tdw/default_rpin.yaml' # default config
    args.gpus = '0'
    args.output = '' # Not used any longer
    #args.init = '' # For restart -> check for tar file
    pprint(vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        num_gpus = torch.cuda.device_count()
    else:
        assert NotImplementedError

    # ---- setup config files
    C.merge_from_file(args.cfg) # <- loads default config
    C.SOLVER.BATCH_SIZE *= num_gpus
    C.SOLVER.BASE_LR *= num_gpus
    # change config based on datasets
    C['DATA_ROOT'] = get_data_paths(data_root, datasets)
    subsets = [subset.split("/")[-1] for subset in datasets]
    C['INPUT']['TRAIN_SLICE'], C['INPUT']['VAL_SLICE'] = get_shift_selector(subsets)
    C['INPUT']['TRAIN_NUM'], C['INPUT']['VAL_NUM'] = get_data_len(subsets)
    C['INPUT']['BINARY_LABELS'] = get_binary_labels(subsets)
    if write_feat:
        C.INPUT.PRELOAD_TO_MEMORY = False
    #TODO might want to adjust prediction length to 10 frames only in config
    #TODO Config cannot be frozen for parallel processing
    #C.freeze()
    model_dir = get_model_dir(train_name, seed, base_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    if write_feat:
        test(args, model_dir, test_name)
    else:
        train(args, model_dir, train_name, num_gpus)
    return


def train(args, output_dir, train_name, num_gpus):
    from neuralphys.datasets.tdw import TDWPhys as PyPhys
    # TODO Changed this to 4 + 6 frame prediction
    C['RPIN']['INPUT_SIZE'] = 4
    C['RPIN']['PRED_SIZE_TRAIN'] = 6 #12
    C['RPIN']['PRED_SIZE_TEST'] = 6 #23

    shutil.copy(args.cfg, os.path.join(output_dir, 'config.yaml'))
    shutil.copy(os.path.join('/home/mrowca/workspace/RPIN/neuralphys/models/', C.RPIN.ARCH + '.py'), os.path.join(output_dir, 'arch.py'))

    # ---- setup logger
    logger = setup_logger('RPIN', output_dir)
    print(git_diff_config(args.cfg))

    model = eval(C.RPIN.ARCH + '.Net')()
    model.to(torch.device('cuda'))
    model = torch.nn.DataParallel(
        model, device_ids=list(range(args.gpus.count(',') + 1))
    )

    # ---- setup optimizer, optimizer is not changed
    vae_params = [p for p_name, p in model.named_parameters() if 'vae_lstm' in p_name]
    other_params = [p for p_name, p in model.named_parameters() if 'vae_lstm' not in p_name]
    optim = torch.optim.Adam(
        [{'params': vae_params, 'weight_decay': 0.0}, {'params': other_params}],
        lr=C.SOLVER.BASE_LR,
        weight_decay=C.SOLVER.WEIGHT_DECAY,
    )

    # ---- if resume experiments, use --init ${model_name}
    init_model_path = os.path.join(output_dir, 'ckpt_best.path.tar')
    if os.path.exists(init_model_path):
        logger.info(f'loading pretrained model from {init_model_path}')
        cp = torch.load(init_model_path)
        model.load_state_dict(cp['model'], False)

    # ---- setup dataset in the last, and avoid non-deterministic in data shuffling order
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_set = PyPhys(data_root=C.DATA_ROOT, split='train')
    val_set = PyPhys(data_root=C.DATA_ROOT, split='test')
    kwargs = {'pin_memory': True, 'num_workers': 0} #16
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=C.SOLVER.BATCH_SIZE, shuffle=True, **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1 if C.RPIN.VAE else C.SOLVER.BATCH_SIZE, shuffle=False, **kwargs,
    )
    print(f'size: train {len(train_loader)} / test {len(val_loader)}')

    # ---- setup trainer
    kwargs = {'device': torch.device('cuda'),
              'model': model,
              'optim': optim,
              'train_loader': train_loader,
              'val_loader': val_loader,
              'output_dir': output_dir,
              'logger': logger,
              'num_gpus': num_gpus,
              'max_iters': C.SOLVER.MAX_ITERS}
    trainer = Trainer(**kwargs)

    #try:
    trainer.train()
    #except BaseException:
    #    if len(glob(f"{output_dir}/*.tar")) < 1:
    #        shutil.rmtree(output_dir)
    #    raise


def test(args, model_dir, test_name):
    from neuralphys.evaluator_feat import PredEvaluator
    if 'human' in test_name:
        from neuralphys.datasets.tdw_human import TDWPhys as PyPhys
    else:
        from neuralphys.datasets.tdw_feat import TDWPhys as PyPhys
    C['RPIN']['INPUT_SIZE'] = 10
    C['RPIN']['PRED_SIZE_TRAIN'] = 5
    C['RPIN']['PRED_SIZE_TEST'] = 5

    output_dir = os.path.join(model_dir, 'features', test_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # --- setup data loader
    print('initialize dataset')
    split_name = 'test'
    val_set = PyPhys(data_root=C.DATA_ROOT, split=split_name, test=True)
    batch_size = 1 if C.RPIN.VAE else C.SOLVER.BATCH_SIZE
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=0) #16

    model = eval(args.predictor_arch + '.Net')()
    model.to(torch.device('cuda'))
    model = torch.nn.DataParallel(
        model, device_ids=[0]
    )
    cp = torch.load(os.path.join(model_dir, 'ckpt_best.path.tar'), map_location=f'cuda:0')
    model.load_state_dict(cp['model'])

    tester = PredEvaluator(
        device=torch.device('cuda'),
        val_loader=val_loader,
        num_gpus=1,
        model=model,
        output_dir=output_dir,
        use_old_path=False,
    )
    tester.test()
    return


def objective(args,
        base_dir = '/mnt/fs4/mrowca/hyperopt/rpin',
        ):
    print(args)
    seed, (train_name, train_data), (test_name, test_data) = args

    if test_name:
        write_feat = 'human' if 'human' in test_name else 'train'
        run(train_name=train_name, test_name=test_name, datasets=test_data, seed=seed,
                data_root='', base_dir=base_dir, write_feat=write_feat)
    else:
        run(train_name=train_name, test_name=test_name, datasets=train_data, seed=seed,
                data_root='', base_dir=base_dir, write_feat='')

    status = STATUS_OK
    loss = 0.0
    return {
            'loss': loss,
            'status': STATUS_OK,
            'seed': seed,
            'train_name': train_name,
            'train_data': train_data,
            'test_name': test_name,
            'test_data': test_data,
            'base_dir': base_dir,
            }



if __name__ == '__main__':
    run()
