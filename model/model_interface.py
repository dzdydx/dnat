# -*- coding: utf-8 -*-
# @Time    : 2022-5-17 22:25:34
# @Author  : Liu Wuyang
# @Email   : liuwuyang@whu.edu.cn
# With functions borrowed from https://github.com/neuralaudio/hear-eval-kit

# Copyright 2021 Zhongyang Zhang
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

import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.nn as nn
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
from .common import validate_score_return_type

num_classes = { "esc50": 50, "audioset": 527}

class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, scores, **kargs):
        super().__init__()
        self.num_classes = num_classes[kargs["dataset"]]
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.scores = scores
        self.use_scoring_for_early_stopping = kargs["use_scoring_for_early_stopping"]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, labels, filename = batch
        out = self(x)
        if isinstance(self.loss_function, torch.nn.CrossEntropyLoss):
            loss = self.loss_function(out, torch.argmax(labels.long(), axis=1))
        else:
            loss = self.loss_function(out, labels)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, labels, filename = batch
    #     out = self(x)
    #     loss = self.loss_function(out, labels)
    #     label_digit = labels.argmax(axis=1)
    #     out_digit = out.argmax(axis=1)

    #     correct_num = sum(label_digit == out_digit).cpu().item()

    #     self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log('val_acc', correct_num/len(out_digit),
    #              on_step=False, on_epoch=True, prog_bar=True)

    #     return (correct_num, len(out_digit))

    def validation_step(self, batch, batch_idx):
        x, y, filename = batch
        y_pr = self(x)
        z = {
            "prediction": y_pr,
            "target": y,
            "filename": filename
        }

        return z

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def _flatten_batched_outputs(
        self,
        outputs,  #: Union[torch.Tensor, List[str]],
        keys: List[str],
        dont_stack: List[str] = [],
    ) -> Dict:
        # ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        flat_outputs_default: DefaultDict = defaultdict(list)
        for output in outputs:
            assert set(output.keys()) == set(keys), f"{output.keys()} != {keys}"
            for key in keys:
                flat_outputs_default[key] += output[key]
        flat_outputs = dict(flat_outputs_default)
        for key in keys:
            if key in dont_stack:
                continue
            else:
                flat_outputs[key] = torch.stack(flat_outputs[key])
        return flat_outputs

    def _score_epoch_end(self, name: str, outputs: List[Dict[str, List[Any]]]):
        flat_outputs = self._flatten_batched_outputs(
            outputs, keys=["target", "prediction", "filename"], dont_stack=["filename"]
        )
        target, prediction, filename = (
            flat_outputs[key] for key in ["target", "prediction", "filename"]
        )

        if isinstance(self.loss_function, torch.nn.CrossEntropyLoss):
            loss = self.loss_function(prediction, torch.argmax(target.long(), axis=1))
        else:
            loss = self.loss_function(prediction, target)

        self.log(
            f"{name}_loss",
            loss,
            prog_bar=True,
            logger=True,
        )

        if name == "test" or self.use_scoring_for_early_stopping:
            self.log_scores(
                name,
                score_args=(
                    prediction.detach().cpu().numpy(),
                    target.detach().cpu().numpy(),
                ),
            )

    def validation_epoch_end(self, outputs: List[Dict[str, List[Any]]]):
        self._score_epoch_end("val", outputs)

    def test_epoch_end(self, outputs: List[Dict[str, List[Any]]]):
        self._score_epoch_end("test", outputs)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay, betas=(0.95, 0.999))

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            elif self.hparams.lr_scheduler == 'multistep':
                scheduler = lrs.MultiStepLR(optimizer, [10, 15, 20, 25], gamma=0.5, last_epoch=-1)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'ce':
            self.loss_function = nn.CrossEntropyLoss()
        elif loss == 'bce_logits':
            self.loss_function = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Invalid Loss Type!")
    
    def log_scores(self, name: str, score_args):
        """Logs the metric score value for each score defined for the model"""
        assert hasattr(self, "scores"), "Scores for the model should be defined"
        end_scores = {}
        # The first score in the first `self.scores` is the optimization criterion
        for score in self.scores:
            score_ret = score(*score_args)
            validate_score_return_type(score_ret)
            # If the returned score is a tuple, store each subscore as separate entry
            if isinstance(score_ret, tuple):
                end_scores[f"{name}_{score}"] = score_ret[0][1]
                # All other scores will also be logged
                for (subscore, value) in score_ret:
                    end_scores[f"{name}_{score}_{subscore}"] = value
            elif isinstance(score_ret, float):
                end_scores[f"{name}_{score}"] = score_ret
            else:
                raise ValueError(
                    f"Return type {type(score_ret)} is unexpected. Return type of "
                    "the score function should either be a "
                    "tuple(tuple) or float."
                )

        self.log(
            f"{name}_score", end_scores[f"{name}_{str(self.scores[0])}"], logger=True
        )
        for score_name in end_scores:
            self.log(score_name, end_scores[score_name], prog_bar=True, logger=True)

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        if name == "passt_base384":
            from model.passt import PaSST
            model = PaSST(stride=10, num_classes=self.num_classes, distilled=True, s_patchout_t=40, s_patchout_f=4)

            # load the pre-trained model state dict
            state_dict = torch.load('/mnt/lwy/amu/checkpoints/passt-s-f128-p16-s10-ap.476-swa.pt')
            # load the weights into the transformer
            model.load_state_dict(state_dict)
            self.model = model
        
        elif name == "ast_base384":
            from .ast import ASTModel
            audio_model = ASTModel(label_dim=self.num_classes, fstride=10, tstride=10, input_fdim=128,
                                input_tdim=998, imagenet_pretrain=True,
                                audioset_pretrain=True, model_size='base384')
            self.model = audio_model

        else:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
            try:
                Model = getattr(importlib.import_module(
                    '.'+name, package=__package__), camel_name)
            except:
                raise ValueError(
                    f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
            self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
