import os
import utils
import shutil
import logging
import argparse
import importlib
import torch
import torch.distributed as dist
from datetime import datetime
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import EpochBasedRunner, build_optimizer, load_checkpoint
from mmdet.apis import set_random_seed
from mmdet.core import DistEvalHook, EvalHook
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from loaders.builder import build_dataloader

def main():

    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=False, default='')
    parser.add_argument('--run_name', required=False, default='')
    parser.add_argument('--override', nargs='+', action=DictAction)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmdet3d'] = logging.Logger(__name__, logging.WARNING)

    # you need GPUs
    assert torch.cuda.is_available()

    # determine local_rank and world_size
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(args.world_size)

    local_rank = int(os.environ['LOCAL_RANK'])  # 0, 1, 2, ...
    world_size = int(os.environ['WORLD_SIZE'])  # 1, 2, 3, ...

    if local_rank == 0:

        # resume or start a new run
        if cfgs.resume_from is not None:
            assert os.path.isfile(cfgs.resume_from)
            work_dir = os.path.dirname(cfgs.resume_from)
        else:
            run_name = args.run_name
            if not cfgs.debug and run_name == '':
                run_name = input('Name your run (leave blank for default): ')
            if run_name == '':
                run_name = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

            if 'ZEEKR_WORK_DIR' not in os.environ:
                work_dir = os.path.join('outputs', cfgs.model.type, run_name)
            else:
                work_dir = os.path.join(os.environ['ZEEKR_WORK_DIR'], cfgs.model.type, run_name)
                print("change to ZEEKR_WORK_DIR:", work_dir)

            if os.path.exists(work_dir):  # must be an empty dir
                if input('Path "%s" already exists, overwrite it? [Y/n] ' % work_dir) == 'n':
                    print('Bye.')
                    exit(0)
                shutil.rmtree(work_dir)

            os.makedirs(work_dir, exist_ok=False)

        # init logging, backup code
        utils.init_logging(os.path.join(work_dir, 'train.log'), cfgs.debug)
        utils.backup_code(work_dir)
        logging.info('Logs will be saved to %s' % work_dir)

    else:
        # disable logging on other workers
        logging.root.disabled = True
        work_dir = '/tmp'

    logging.info('Using GPU: %s' % torch.cuda.get_device_name(local_rank))
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        logging.info('Initializing DDP with %d GPUs...' % world_size)
        dist.init_process_group('nccl', init_method='env://')

    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)

    logging.info('Loading training set from %s' % cfgs.dataset_root)

    # * build training dataset
    train_dataset = build_dataset(cfgs.data.train)

    # * build training dataloader
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=cfgs.batch_size,
        workers_per_gpu=cfgs.data.workers_per_gpu,  
        num_gpus=world_size,
        dist=world_size > 1,
        shuffle=True,
        seed=0,
    )
    
    logging.info('Loading validation set from %s' % cfgs.dataset_root)
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=world_size,
        dist=world_size > 1,
        shuffle=False
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    
    model = build_model(cfgs.model)
    model.init_weights()
    model.cuda()
    model.train()

    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logging.info('Trainable parameters: %d (%.1fM)' % (n_params, n_params / 1e6))
    # logging.info('Batch size per GPU: %d' % (cfgs.batch_size // world_size))
    logging.info('Batch size per GPU: %d' % (cfgs.batch_size))

    if os.path.isfile(args.weights):
        logging.info('Loading checkpoint from %s' % args.weights)
        load_checkpoint(
            model, args.weights, map_location='cuda', strict=True,
            logger=logging.Logger(__name__, logging.ERROR)
        )

    if world_size > 1:
        model = MMDistributedDataParallel(model, [local_rank])
        # model._set_static_graph()
    else:
        model = MMDataParallel(model, [0])

    logging.info('Creating optimizer: %s' % cfgs.optimizer.type)
    optimizer = build_optimizer(model, cfgs.optimizer)

    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=work_dir,
        logger=logging.root,
        max_epochs=cfgs.total_epochs,
        meta=dict(),
    )

    # init some hooks
    runner.register_lr_hook(cfgs.lr_config)                         # learning rate scheduler {'policy': 'step', 'warmup': 'linear', 'warmup_iters': 500, 'warmup_ratio': 0.3333333333333333, 'by_epoch': True, 'step': [22, 24], 'gamma': 0.2}
    runner.register_optimizer_hook(cfgs.optimizer_config)           # optimizer config {'grad_clip': {'max_norm': 35, 'norm_type': 2}}
    runner.register_checkpoint_hook(cfgs.checkpoint_config)         # ckpt config {'interval': 1, 'max_keep_ckpts': 1}
    runner.register_logger_hooks(cfgs.log_config)                   # log config
    runner.register_timer_hook(dict(type='IterTimerHook'))          # timer hook
    runner.register_custom_hooks(dict(type='DistSamplerSeedHook'))  # fix seed for distributed sampler

    if cfgs.eval_config['interval'] > 0:
        if world_size > 1:
            runner.register_hook(DistEvalHook(val_loader, interval=cfgs.eval_config['interval'], gpu_collect=True))
        else:
            runner.register_hook(EvalHook(val_loader, interval=cfgs.eval_config['interval']))

    if cfgs.resume_from is not None:
        logging.info('Resuming from %s' % cfgs.resume_from)
        runner.resume(cfgs.resume_from)

    elif cfgs.load_from is not None:
        logging.info('Loading checkpoint from %s' % cfgs.load_from)
        if cfgs.revise_keys is not None:            # [('backbone', 'img_backbone')]
            load_checkpoint(
                model, cfgs.load_from, map_location='cpu',
                revise_keys=cfgs.revise_keys,        # [('backbone', 'img_backbone')]
                strict=False
            )
        else:
            load_checkpoint(
                model, cfgs.load_from, map_location='cpu',
                strict=False
            )
    runner.run([train_loader], [('train', 1)])


if __name__ == '__main__':
    main()