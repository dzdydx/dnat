# -*- coding: utf-8 -*-
# @Time    : 2022-5-13 21:44:52
# @Author  : Liu Wuyang
# @Email   : liuwuyang@whu.edu.cn
# Modified from https://github.com/neuralaudio/hear-eval-kit

import torch
import pytorch_lightning as pl
from torch import nn
import torchinfo
from einops import reduce, rearrange
from hear21passt.base import load_model, get_scene_embeddings, get_timestamp_embeddings
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

PARAM_GRID = {
    "hidden_layers": 2,
    # "hidden_layers": [0, 1, 2],
    # "hidden_layers": [1, 2, 3],
    "hidden_dim": 1024,
    # "hidden_dim": [256, 512, 1024],
    # "hidden_dim": [1024, 512],
    # Encourage 0.5
    "dropout": 0.1,
    # "dropout": [0.1, 0.5],
    # "dropout": [0.1, 0.3],
    # "dropout": [0.1, 0.3, 0.5],
    # "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    # "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "lr": 1e-3,
    # "lr": [3.2e-3, 1e-3, 3.2e-4, 1e-4, 3.2e-5, 1e-5],
    # "lr": [1e-2, 3.2e-3, 1e-3, 3.2e-4, 1e-4],
    # "lr": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    "patience": 20,
    "max_epochs": 500,
    # "max_epochs": [500, 1000],
    "check_val_every_n_epoch": 3,
    # "check_val_every_n_epoch": [1, 3, 10],
    "batch_size": 32,
    # "batch_size": [1024, 2048],
    # "batch_size": [256, 512, 1024],
    # "batch_size": [256, 512, 1024, 2048, 4096, 8192],
    "hidden_norm": torch.nn.BatchNorm1d,
    # "hidden_norm": [torch.nn.Identity, torch.nn.BatchNorm1d, torch.nn.LayerNorm],
    "norm_after_activation": False,
    # "norm_after_activation": [False, True],
    "embedding_norm": torch.nn.Identity,
    # "embedding_norm": [torch.nn.Identity, torch.nn.BatchNorm1d],
    # "embedding_norm": [torch.nn.Identity, torch.nn.BatchNorm1d, torch.nn.LayerNorm],
    "initialization": torch.nn.init.xavier_uniform_,
    "optim": torch.optim.Adam,
    # "optim": [torch.optim.Adam, torch.optim.SGD],
}

class OneHotToCrossEntropyLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # One and only one label per class
        assert torch.all(
            torch.sum(y, dim=1) == torch.ones(y.shape[0], device=self.device)
        )
        y = y.argmax(dim=1)
        return self.loss(y_hat, y)

class FullyConnectedPrediction(torch.nn.Module):
    def __init__(self, nfeatures: int, class_num: int, embedding_type: str, prediction_type: str, conf: Dict):
        super().__init__()
        hidden_modules: List[torch.nn.Module] = []
        curdim = nfeatures
        last_activation = "linear"
        if conf["hidden_layers"]:
            for i in range(conf["hidden_layers"]):
                linear = torch.nn.Linear(curdim, conf["hidden_dim"])
                conf["initialization"](
                    linear.weight,
                    gain=torch.nn.init.calculate_gain(last_activation),
                )
                hidden_modules.append(linear)
                if not conf["norm_after_activation"]:
                    hidden_modules.append(conf["hidden_norm"](conf["hidden_dim"]))
                hidden_modules.append(torch.nn.Dropout(conf["dropout"]))
                hidden_modules.append(torch.nn.ReLU())
                if conf["norm_after_activation"]:
                    hidden_modules.append(conf["hidden_norm"](conf["hidden_dim"]))
                curdim = conf["hidden_dim"]
                last_activation = "relu"

            self.hidden = torch.nn.Sequential(*hidden_modules)
        else:
            self.hidden = torch.nn.Identity()  # type: ignore
        self.projection = torch.nn.Linear(curdim, class_num)

        self.embedding_type = embedding_type

        conf["initialization"](
            self.projection.weight, gain=torch.nn.init.calculate_gain(last_activation)
        )
        if prediction_type == "multilabel":
            self.activation: torch.nn.Module = torch.nn.Sigmoid()
        elif prediction_type == "multiclass":
            self.activation = torch.nn.Softmax()
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}")

    def forward_logit(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.projection(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding_type == "scene":
            x = self.forward_logit(x)
            x = self.activation(x)
        elif self.embedding_type == "timestamp":
            bsz = x.shape[0]
            x = rearrange(x, 'b t f -> (b t) f')
            x = self.forward_logit(x)
            x = self.activation(x)
            x = rearrange(x, '(b t) f -> b t f', b=bsz)
            x = x.mean(axis=1)
        return x


class PasstWrapper(nn.Module):
    def __init__(
        self,
        passt_path: str,
        nfeatures: int,
        class_num: int,
        embedding_type: str,
        prediction_type: str,
        conf: Dict = PARAM_GRID,
        model_options: Optional[Dict[str, Any]] = None,
    ):
        super(PasstWrapper, self).__init__()
        if model_options is None:
            model_options = {}
        
        self.embedding_type = embedding_type

        if passt_path is not None:
            self.model = load_model(passt_path, **model_options)
        else:
            self.model = load_model(**model_options)
        
        # Load Predictor
        if self.embedding_type == "scene":
            self.predictor = FullyConnectedPrediction(
                nfeatures, class_num, prediction_type, conf
            )
            torchinfo.summary(self.predictor, input_size=(32, nfeatures))
        elif self.embedding_type == "timestamp":
            self.predictor = FullyConnectedPrediction(
                nfeatures, class_num, prediction_type, conf
            )
            torchinfo.summary(self.predictor, input_size=(32, 51, nfeatures))
        

    def get_scene_embedding(
        self, audio: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            embeddings = get_scene_embeddings(audio, self.model)
            return embeddings.detach()

    def get_timestamp_embedding(
        self, audio: torch.Tensor
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        with torch.no_grad():
            embeddings, timestamps = get_timestamp_embeddings(audio, self.model)
            embeddings = embeddings.detach()
            timestamps = timestamps.detach()
            return embeddings, timestamps
    
    def forward(self, x):
        if self.embedding_type == 'scene':
            x = self.get_scene_embedding(x)
            x = self.predictor(x)
        elif self.embedding_type == 'timestamp':
            embeddings, timestamps = self.get_scene_embedding(x)
            x = self.predictor(x)
        return x
