import time
from torch import nn
import torch
import warnings

from src.model.perception import Perception
from src.backbones.vae import CVAE
from src.model.diversity import DiversityDiagAE
from src.configs import DataName, ModelType


class MTG(nn.Module):
    def __init__(self, cfgs):
        super(MTG, self).__init__()
        self.cfg = cfgs

        self.perception = Perception(self.cfg.perception)

        self.model_type = self.cfg.dlow.model_type
        self.paths_num = self.cfg.dlow.paths_num
        if self.model_type == ModelType.cvae:
            self.generator = CVAE(self.cfg.dlow, activation_func=self.cfg.dlow.activation_func)
        elif self.model_type == ModelType.dlowae:
            self.generator = DiversityDiagAE(self.cfg.dlow, activation_func=self.cfg.dlow.activation_func)
        else:
            raise Exception("model type is not defined")

        if self.cfg.perception.fix_perception:
            self.set_perception_fixed()

    def set_perception_fixed(self):
        for param in self.perception.parameters():
            param.requires_grad = False

    def forward(self, input_dict):
        output = {DataName.path: input_dict[DataName.path],
                  DataName.png: input_dict[DataName.png],
                  DataName.last_poses: input_dict[DataName.last_poses]}
        if DataName.all_paths in input_dict.keys():
            output.update({DataName.all_paths: input_dict[DataName.all_paths]})

        observation = self.perception(input_dict)

        if self.model_type == ModelType.cvae:
            waypoints, mu, logvar = self.generator(observation)
            output.update({DataName.mu: mu, DataName.logvar: logvar})
        elif self.model_type == ModelType.dlowae:
            waypoints, A, b, mu, logvar, scores = self.generator(observation)
            output.update({DataName.A: A,
                           DataName.b: b,
                           DataName.mu: mu,
                           DataName.logvar: logvar,
                           DataName.scores: scores})
        else:
            raise Exception("model type is not defined")
        output.update({DataName.y_hat: waypoints})
        return output

    def sample_forward(self, input_dict, N):
        output = {DataName.path: input_dict[DataName.path],
                  DataName.last_poses: input_dict[DataName.last_poses]}
        observation = self.perception(input_dict)
        if self.model_type == ModelType.cvae:
            waypoints, mu, logvar = self.generator.sample_forward(observation, N)
            output.update({DataName.mu: mu, DataName.logvar: logvar})
        elif self.model_type == ModelType.dlowae:
            waypoints, A, b, mu, logvar = self.generator(observation)
            output.update({DataName.A: A, DataName.b: b, DataName.mu: mu, DataName.logvar: logvar})
        else:
            raise Exception("model type is not defined")
        output.update({DataName.y_hat: waypoints})
        return output

    def deterministic_forward(self, input_dict):
        output = {}
        if DataName.path in input_dict.keys():
            output.update({DataName.path: input_dict[DataName.path],
                      DataName.last_poses: input_dict[DataName.last_poses]})
        if DataName.all_paths in input_dict.keys():
            output.update({DataName.all_paths: input_dict[DataName.all_paths],
                            DataName.png: input_dict[DataName.png],
                           DataName.camera: input_dict[DataName.camera]})
        observation = self.perception(input_dict)
        if self.model_type == ModelType.cvae:
            waypoints, mu, logvar = self.generator.sample_forward(observation, N=self.paths_num)
            output.update({DataName.mu: mu, DataName.logvar: logvar})
        elif self.model_type == ModelType.dlowae:
            waypoints, A, b, mu, logvar = self.generator.deterministic_forward(observation)
            output.update({DataName.A: A, DataName.b: b, DataName.mu: mu, DataName.logvar: logvar})
        else:
            raise Exception("model type is not defined")
        output.update({DataName.y_hat: waypoints})
        return output

    def experiment_forward(self, input_dict, N):
        output = {DataName.path: input_dict[DataName.path],
                  DataName.last_poses: input_dict[DataName.last_poses]}
        observation = self.perception(input_dict)
        if self.model_type == ModelType.cvae:
            waypoints, mu, logvar = self.generator.sample_forward(observation, N)
            output.update({DataName.mu: mu, DataName.logvar: logvar})
        elif self.model_type == ModelType.dlowae:
            waypoints, A, b, mu, logvar = self.generator.deterministic_forward(observation)
            output.update({DataName.A: A, DataName.b: b, DataName.mu: mu, DataName.logvar: logvar})
        else:
            raise Exception("model type is not defined")