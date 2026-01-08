# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import shutil
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmrotate.utils import register_all_modules


wekan_config = '../configs/rotated_fcos/rotated-fcos-le90_r101_fpn_3x_wavekan-RSAR.py'
wekan_resume = True
wekan_name = 'WEAttention-KAN-T05T05F'

wekan_loss_RIoU = dict(type='RotatedIoULoss', loss_weight=0.5) # None
wekan_loss_GD = dict(type='GDLoss', loss_type='gwd', fun='log1p', tau=1.0, alpha=1.0, loss_weight=0.5)
wekan_loss_L1 = None # None
wekan_KAN = True
wekan_Atten = 'spatial_channel' # spatial_channel Cross Gate
wekan_Wave_J = 1
wekan_loss_kan = dict(type='KANIdenty', loss_weight=0.0)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    args.config = wekan_config
    args.resume = wekan_resume
    # register all modules in mmdet into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.model.bbox_head.loss_bbox = wekan_loss_RIoU
    cfg.model.bbox_head.loss_GD = wekan_loss_GD
    cfg.model.bbox_head.loss_L1 = wekan_loss_L1
    cfg.model.bbox_head.kan = wekan_KAN
    cfg.model.bbox_head.loss_kan = wekan_loss_kan
    cfg.model.backbone.atten = wekan_Atten
    cfg.model.backbone.Wave_J = wekan_Wave_J

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir + '/' + wekan_name
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0]) + '/' + wekan_name
    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    cfg.resume = args.resume
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    shutil.copy2(__file__, osp.join(cfg.work_dir, os.path.basename(__file__)))
    #runner.train()
    runner.train()


if __name__ == '__main__':
    main()
