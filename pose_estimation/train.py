from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.distributed as dist
import paddle.vision.transforms as transforms
from visualdl import LogWriter

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer,get_lr_scheduler
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import resume

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--batch_size',
                        help='batch_size',
                        type=int)
    parser.add_argument('--resume_model',
                        help='resume path',
                        default=None,
                        type=str)
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = args.batch_size

def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
        final_output_dir)

    writer = LogWriter(tb_log_dir) # visual log
    writer_dict = {
        'writer': writer,
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    nranks = paddle.distributed.ParallelEnv().nranks
    if nranks>1:
        dist.init_parallel_env()
        model = paddle.DataParallel(model)

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        transforms.Compose([
            normalize,
        ])
    )
    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            normalize,
        ])
    )

    train_sampler = paddle.io.DistributedBatchSampler(dataset=train_dataset,batch_size=config.TRAIN.BATCH_SIZE,shuffle=config.TRAIN.SHUFFLE,drop_last=False)
    train_loader = paddle.io.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.WORKERS,
        use_shared_memory=False,
    )
    val_sampler = paddle.io.DistributedBatchSampler(dataset=valid_dataset,batch_size=config.TEST.BATCH_SIZE,shuffle=False,drop_last=False)
    valid_loader = paddle.io.DataLoader(
        valid_dataset,
        batch_sampler=val_sampler,
        num_workers=config.WORKERS,
        use_shared_memory=False,
    )

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    )

    lr_scheduler = get_lr_scheduler(config,len(train_loader))
    optimizer = get_optimizer(lr_scheduler,config, model)

    start_epoch = config.TRAIN.BEGIN_EPOCH
    if args.resume_model is not None:
        start_epoch = resume(model, optimizer, args.resume_model)

    # train
    best_perf = 0.0
    best_model = False
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        if epoch < start_epoch:
            continue
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict, config.TRAIN.END_EPOCH)


        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, valid_dataset, model,
                                  criterion, final_output_dir, tb_log_dir,
                                  writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pdparams')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    paddle.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
