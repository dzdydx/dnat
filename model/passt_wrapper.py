# -*- coding: utf-8 -*-
# @Time    : 2022-5-13 21:44:52
# @Author  : Liu Wuyang
# @Email   : liuwuyang@whu.edu.cn
# Modified from https://github.com/neuralaudio/hear-eval-kit

import torch
from torch import nn
from hear21passt.base import load_model, get_scene_embeddings, get_timestamp_embeddings

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
    """
    for ESC-50 :
        nfeatures = 
        nlabels = 50,
        prediction_type = 'multiclass',
        conf = conf
    """
    def __init__(self, nfeatures: int, nlabels: int, prediction_type: str, conf: Dict):
        super().__init__()
        """
        conf = {
            hidden_layers: int,
            hidden_dim: int,
            initialization: Function,
            norm_after_activation: bool,
            dropout: float,
        }
        """
        hidden_modules: List[torch.nn.Module] = []
        curdim = nfeatures
        # Honestly, we don't really know what activation preceded
        # us for the final embedding.
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
        self.projection = torch.nn.Linear(curdim, nlabels)

        conf["initialization"](
            self.projection.weight, gain=torch.nn.init.calculate_gain(last_activation)
        )
        self.logit_loss: torch.nn.Module
        if prediction_type == "multilabel":
            self.activation: torch.nn.Module = torch.nn.Sigmoid()
            self.logit_loss = torch.nn.BCEWithLogitsLoss()
        elif prediction_type == "multiclass":
            self.activation = torch.nn.Softmax()
            self.logit_loss = OneHotToCrossEntropyLoss()
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}")

    def forward_logit(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.projection(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_logit(x)
        x = self.activation(x)
        return x


class PasstWrapper(nn.Module):
    """
    Args:
        module_name: the import name for the embedding module
        model_path: location to load the model from
    """

    def __init__(
        self,
        nfeatures: int,
        label_to_idx: Dict[str, int],
        nlabels: int,
        prediction_type: str,
        scores: List[ScoreFunction],
        conf: Dict,
        use_scoring_for_early_stopping: bool = True,
        model_options: Optional[Dict[str, Any]] = None,
    ):

        if model_options is None:
            model_options = {}

        # Load the model using the model weights path if they were provided
        if model_path is not None:
            self.model = load_model(model_path, **model_options)
        else:
            self.model = load_model(**model_options)
        
        # Load Predictor
        self.predictor = FullyConnectedPrediction(
            nfeatures, nlabels, prediction_type, conf
        )
        torchinfo.summary(self.predictor, input_size=(32, nfeatures))

        #! Load score matrics later
        # self.scores = scores

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
            # flake8: noqa
            embeddings, timestamps = get_timestamp_embeddings(audio, self.model)
            embeddings = embeddings.detach()
            timestamps = timestamps.detach()
            return embeddings, timestamps
    
    def forward(self, x):
        x = self.get_scene_embedding(x)
        x = self.predictor(x)
        return x
