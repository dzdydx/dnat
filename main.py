# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import pandas as pd
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from model import MInterface
from data import DInterface
from score import available_scores, label_vocab_as_dict, SelectedMeanAveragePrecision

def load_scores(args):
    if args.dataset == 'esc50':
        evaluations = ["top1_acc", "mAP", "d_prime", "aucroc"]
    elif args.dataset == 'audioset':
        evaluations = ["mAP", "aucroc", "d_prime"]

    label_vocab = pd.read_csv(args.label_csv)
    nlabels = len(label_vocab)
    assert nlabels == label_vocab["idx"].max() + 1
    label_to_idx = label_vocab_as_dict(label_vocab, key="label", value="idx")

    scores = [
        available_scores[score](label_to_idx=label_to_idx)
        for score in evaluations
    ]
    if args.selected_mAP:
        scores.append(SelectedMeanAveragePrecision(label_to_idx=label_to_idx))
    return scores

def load_callbacks(args):
    callbacks = []
    scores = args.scores
    if args.use_scoring_for_early_stopping:
        # First score is the target
        target_score = f"val_{str(scores[0])}"
        if scores[0].maximize:
            mode = "max"
        else:
            mode = "min"
    else:
        # This loss is much faster, but will give poorer scores
        target_score = "val_loss"
        mode = "min"

    callbacks.append(plc.EarlyStopping(
        monitor=target_score,
        mode=mode,
        patience=20,
        min_delta=0.001,
        check_on_train_epoch_end=False,
        verbose=False,
    ))

    # Model check point cfg
    if args.dataset == 'esc50':
        callbacks.append(plc.ModelCheckpoint(
            monitor=target_score,
            filename='best-{epoch:02d}-{val_top1_acc:.3f}',
            save_top_k=1,
            mode=mode,
            save_last=True
        ))
    elif args.dataset == 'audioset':
        callbacks.append(plc.ModelCheckpoint(
            monitor=target_score,
            filename='best-{epoch:02d}-{val_mAP:.3f}',
            save_top_k=1,
            mode=mode,
            save_last=True
        ))


    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='step'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)

    # Load scores & callbacks
    args.use_scoring_for_early_stopping = True
    if args.dataset == 'audioset' and args.audioset_train == 'full':
        args.val_check_interval = 0.1
        
    args.scores = load_scores(args)
    args.callbacks = load_callbacks(args)

    data_module = DInterface(**vars(args))
    model = MInterface(**vars(args))

    # # If you want to change the logger's saving folder
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.model_name)
    args.logger = logger

    trainer = Trainer.from_argparse_args(args, accelerator='gpu', devices=1)

    if args.ckpt_path is None:
        if args.mode == "train":
            trainer.fit(model, data_module)
        result = trainer.test(model, data_module)
        # print(result)
    else:
        if args.mode == "train":
            trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
        result = trainer.test(model, data_module, ckpt_path=args.ckpt_path)
        # print(result)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Mode control
    parser.add_argument('mode', choices=['train', 'eval'])

    # Basic Training Control
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epochs', default=100, type=int)

    # LR Scheduler
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine', 'multistep', 'exp_lin', 'cos_cyc'], type=str)
    parser.add_argument('--lr_decay_steps', type=int)
    parser.add_argument('--lr_decay_rate', type=float)
    parser.add_argument('--lr_decay_min_lr', type=float)

    # Restart Control
    parser.add_argument('--ckpt_path', default=None, type=str)
    parser.add_argument('--pretrain_model', default=None, type=str)

    # Training Info
    parser.add_argument('--dataset', default='esc50', type=str)
    parser.add_argument('--dataset_type', default='audio_tagging_dataset', type=str)
    parser.add_argument('--audioset_train', default='balanced', type=str)

    parser.add_argument('--mixup_strategy', default='vanilla', type=str)
    parser.add_argument('--mixup_ratio', default=0.5, type=float)
    parser.add_argument('--use_weighted_mixup', action="store_true")

    parser.add_argument('--sample_rate', default=32000, type=int)
    parser.add_argument('--train_json', type=str)
    parser.add_argument('--val_json', type=str)
    parser.add_argument('--test_json', type=str)
    parser.add_argument('--label_csv', type=str)
    
    parser.add_argument('--loss', default='bce', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--log_dir', default='log', type=str)
    
    # Model Hyperparameters
    parser.add_argument('--model_name', default='standard_net', type=str)
    parser.add_argument('--pretrained', action="store_true")
    # parser.add_argument('--hid', default=64, type=int)
    # parser.add_argument('--block_num', default=8, type=int)
    # parser.add_argument('--in_channel', default=1, type=int)
    # parser.add_argument('--layer_num', default=5, type=int)

    # PASST wrapper params
    # parser.add_argument('--passt_path', type=str)
    # parser.add_argument('--nfeatures', default=1295, type=int)
    # parser.add_argument('--embedding_type', default="scene", type=str)
    # parser.add_argument('--prediction_type', default="multiclass", type=str)

    # Other
    parser.add_argument('--selected_mAP', action="store_true")

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(max_epochs=100)

    args = parser.parse_args()
    
    main(args)
