import copy
import time
import os

import torch
from tqdm import tqdm
import numpy as np
import torch as th
import cv2
import os.path as osp
from datetime import datetime
from src.configs import cfg, LossDictKeys, DataName
from src.utils.functions import get_device, to_device, release_cuda
from src.utils.logger import SummaryBoard, configure_logger
from src.data_loader import train_eval_data_loader
from src.dlow.model import DLOW
from src.loss import LossEvaluation


class Trainer:
    def __init__(self, cfgs: cfg):
        self.device = get_device(device=cfgs.device)

        self.snapshot = cfgs.load_snapshot
        self.name = cfgs.name

        self.data_loader, self.val_loader = train_eval_data_loader(cfg=cfgs.data)
        self.model = DLOW(cfgs=cfgs.model).to(self.device)

        self.cfgs = cfgs.training
        self.w_eval = self.cfgs.w_eval
        self.max_epoch = self.cfgs.max_epoch
        self.max_iteration_per_epoch = self.cfgs.max_iteration_per_epoch
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.cfgs.lr, weight_decay=self.cfgs.weight_decay)
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer, self.cfgs.lr_decay_steps,
                                                      gamma=self.cfgs.lr_decay)

        self.iteration = 0
        self.epoch = 0
        self.training = True
        self.grad_acc_steps = self.cfgs.grad_acc_steps
        self.best_loss = np.inf

        # loss function, evaluator
        self.loss_func = LossEvaluation(cfg=cfgs.loss_eval).to(self.device)
        self.evaluator = LossEvaluation(cfg=cfgs.loss_eval).to(self.device)

        self.log_steps = cfgs.logger.log_steps
        self.log_summary = SummaryBoard()
        self.logger = configure_logger(verbose=cfgs.logger.verbose,
                                       tensorboard_log=cfgs.logger.log_name,
                                       tb_log_name=self.name + "-" + datetime.now().strftime("%m-%d-%Y-%H-%M"),
                                       reset_num_timesteps=cfgs.logger.reset_num_timesteps)

    # TODO: check later
    def save_snapshot(self, filename):
        model_state_dict = self.model.state_dict()

        # save model
        state_dict = {'model': model_state_dict}
        th.save(state_dict, filename)
        self.logger.info('Model saved to "{}"'.format(filename))

        # save snapshot
        state_dict['epoch'] = self.epoch
        state_dict['iteration'] = self.iteration
        snapshot_filename = osp.join(str(self.name) + 'snapshot.pth.tar')
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        th.save(state_dict, snapshot_filename)
        self.logger.info('Snapshot saved to "{}"'.format(snapshot_filename))

    # TODO: check later
    def load_snapshot(self, snapshot):
        self.logger.info('Loading from "{}".'.format(snapshot))
        state_dict = th.load(snapshot, map_location=th.device('cpu'))

        # Load model
        model_dict = state_dict['model']
        self.model.load_state_dict(model_dict, strict=False)

        # log missing keys and unexpected keys
        snapshot_keys = set(model_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if len(missing_keys) > 0:
            message = f'Missing keys: {missing_keys}'
            self.logger.error(message)
        if len(unexpected_keys) > 0:
            message = f'Unexpected keys: {unexpected_keys}'
            self.logger.error(message)
        self.logger.info('Model has been loaded.')

        # Load other attributes
        if 'epoch' in state_dict:
            self.epoch = state_dict['epoch']
            self.logger.info('Epoch has been loaded: {}.'.format(self.epoch))
        if 'iteration' in state_dict:
            self.iteration = state_dict['iteration']
            self.logger.info('Iteration has been loaded: {}.'.format(self.iteration))
        if 'optimizer' in state_dict and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state_dict['optimizer'])
                self.logger.info('Optimizer has been loaded.')
            except:
                print("doesn't load optimizer")
        if 'scheduler' in state_dict and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state_dict['scheduler'])
                self.logger.info('Scheduler has been loaded.')
            except:
                print("doesn't load scheduler")

    def set_train_mode(self):
        self.training = True
        self.model.train()
        th.set_grad_enabled(True)

    def set_eval_mode(self):
        self.training = False
        self.model.eval()
        th.set_grad_enabled(False)

    def optimizer_step(self, iteration):
        if iteration % self.grad_acc_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def step(self, data_dict):
        data_dict = to_device(data_dict, device=self.device)
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict)
        # result_dict = self.evaluator(output_dict)
        # loss_dict.update(result_dict)
        return output_dict, loss_dict

    def update_logger(self, result_dict: dict):
        for key, value in result_dict.items():
            self.logger.record("train/{}".format(key), value=value)

    def run_epoch(self):
        self.optimizer.zero_grad()
        # self.data_loader.reset()
        last_time = time.time()
        for iteration, data_dict in enumerate(tqdm(self.data_loader, desc="Epoch {}".format(self.epoch))):
            if iteration % self.max_iteration_per_epoch == 0 and iteration != 0:
                break
            self.iteration += 1

            # data_time = time.time()
            # print("dt: {:.4f}".format(data_time - last_time), end=", ")

            # step_time_start = time.time()
            output_dict, result_dict = self.step(data_dict=data_dict)
            # step_time = time.time()
            # print("st: {:.4f}".format(step_time - step_time_start), end=", ")
            th.cuda.empty_cache()
            # self._display_output(output_dict=output_dict, data_dict=data_dict, iteration=iteration,
            #                      root_path="/home/jing/Documents/gn/global_nav/test/evaluation/training")
            result_dict[LossDictKeys.loss].backward()
            # for name, param in self.model.named_parameters():
            #     print(name, param.grad)

            # optimize_time_start = time.time()
            self.optimizer_step(iteration + 1)
            # optimize_time = time.time()
            # print("ot: {:.4f}".format(optimize_time - optimize_time_start))

            result_dict = release_cuda(result_dict)
            self.log_summary.update_from_result_dict(result_dict)
            # self.log_summary.update("step_time", step_time - start_time)
            # self.log_summary.update("opt_time", optimize_time - step_time)
            # self.log_summary.update("total_time", optimize_time - start_time)
            if iteration % self.log_steps == 0:
                summary_dict = self.log_summary.summary()
                self.logger.record_dict(summary_dict, prefix="train/")
                self.logger.record("train/iterations", iteration, exclude="tensorboard")
                self.logger.record("train/learning_rate", self.scheduler.get_last_lr())
                self.logger.dump(step=self.iteration)
                self.log_summary.reset_all()
            th.cuda.empty_cache()
            # last_time = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

        os.makedirs("./models/{}".format(self.name), exist_ok=True)
        self.save_snapshot(f'models/{self.name}/last.pth.tar')

    def _write_png(self, local_map=None, center=None, targets=None, paths=None, path=None,
                   others=None, file="/home/jing/Documents/gn/global_nav/test/test_local_map.png"):
        dis = 2
        if len(local_map.shape) == 2:
            local_map_fig = np.repeat(local_map[:, :, np.newaxis], 3, axis=2) * 255
        else:
            local_map_fig = copy.deepcopy(local_map)

        if center is not None:
            assert center.shape[0] == 2 and len(center.shape) == 1, "path should be 2"
            all_points = []
            for x in range(-dis, dis, 1):
                for y in range(-dis, dis, 1):
                    all_points.append(center + np.array([x, y]))
            all_points = np.stack(all_points).astype(int)
            local_map_fig[all_points[:, 0], all_points[:, 1], 2] = 255
            local_map_fig[all_points[:, 0], all_points[:, 1], 0] = 0
            local_map_fig[all_points[:, 0], all_points[:, 1], 1] = 0
        if targets is not None:
            all_points = []
            for x in range(-dis, dis, 1):
                for y in range(-dis, dis, 1):
                    all_points.append(targets + np.array([x, y]))
            if len(targets.shape) == 2:
                all_points = np.concatenate(all_points, axis=0).astype(int)
            else:
                all_points = np.stack(all_points, axis=0).astype(int)
            local_map_fig[all_points[:, 0], all_points[:, 1], 1] = 255
            local_map_fig[all_points[:, 0], all_points[:, 1], 0] = 0
            local_map_fig[all_points[:, 0], all_points[:, 1], 2] = 0
        if others is not None:
            assert others.shape[1] == 2 and len(others.shape) == 2, "path should be Nx2"
            all_points = []
            for x in range(-dis, dis, 1):
                for y in range(-dis, dis, 1):
                    all_points.append(others + np.array([x, y]))
            all_points = np.clip(np.concatenate(all_points, axis=0).astype(int), 0, local_map_fig.shape[0] - 1)
            local_map_fig[all_points[:, 0], all_points[:, 1], 1] = 255
            local_map_fig[all_points[:, 0], all_points[:, 1], 0] = 255
            local_map_fig[all_points[:, 0], all_points[:, 1], 2] = 0
        if path is not None:
            assert path.shape[1] == 2 and len(path.shape) == 2 and path.shape[0] >= 2, "path should be Nx2"
            all_pts = path
            all_pts = np.concatenate((all_pts + np.array([0, -1], dtype=int), all_pts + np.array([1, 0], dtype=int),
                                      all_pts + np.array([-1, 0], dtype=int), all_pts + np.array([0, 1], dtype=int),
                                      all_pts), axis=0)
            all_pts = np.clip(all_pts, 0, local_map_fig.shape[0] - 1)
            local_map_fig[all_pts[:, 0], all_pts[:, 1], 0] = 255
            local_map_fig[all_pts[:, 0], all_pts[:, 1], 1] = 0
            local_map_fig[all_pts[:, 0], all_pts[:, 1], 2] = 255
        if paths is not None:
            for path in paths:
                if len(path) == 1 or np.any(path[0] == np.inf):
                    continue
                path = np.asarray(path, dtype=int)
                assert path.shape[1] == 2 and len(path.shape) == 2 and path.shape[0] >= 2, "path should be Nx2"
                all_pts = np.concatenate((path + np.array([0, -1], dtype=int), path + np.array([1, 0], dtype=int),
                                          path + np.array([-1, 0], dtype=int), path + np.array([0, 1], dtype=int),
                                          path), axis=0)
                all_pts = np.clip(all_pts, 0, local_map_fig.shape[0] - 1)
                local_map_fig[all_pts[:, 0], all_pts[:, 1], 0] = 255
                local_map_fig[all_pts[:, 0], all_pts[:, 1], 1] = 0
                local_map_fig[all_pts[:, 0], all_pts[:, 1], 2] = 255
        cv2.imwrite(file, local_map_fig)
        return local_map_fig

    def _display_output(self, output_dict, data_dict, iteration, local_map_resolution=0.1, local_map_threshold=300,
                        root_path="/home/jing/Documents/gn/global_nav/test/evaluation/training"):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        y_hat = output_dict[DataName.y_hat]  # BxNx2
        for i in range(len(y_hat)):
            file_name = str(int(data_dict[DataName.local_map][i].detach().cpu().numpy()[0])) + ".png"
            local_map = cv2.imread(os.path.join(self.data_loader.dataset.root, self.data_loader.dataset.figures, file_name))
            local_map = local_map[int(local_map.shape[0] / 2 - local_map_threshold):
                                  int(local_map.shape[0] / 2 + local_map_threshold),
                        int(local_map.shape[1] / 2 - local_map_threshold):
                        int(local_map.shape[1] / 2 + local_map_threshold)]
            center = np.array([local_map.shape[0] / 2.0, local_map.shape[1] / 2.0], dtype=int)

            scan_pixels = np.clip(np.floor(data_dict[DataName.scan][i].detach().cpu().numpy()[:, :2] /
                                           local_map_threshold).astype(int) + center, 0, local_map.shape[0] - 1)
            if len(y_hat.shape) == 3:
                points = torch.cumsum(y_hat[i], dim=0).detach().cpu().numpy()
                pixels = np.clip(np.floor(points / local_map_resolution).astype(int) + center, 0,
                                 local_map.shape[0] - 1)
                self._write_png(local_map=local_map, center=center, path=pixels, others=scan_pixels,
                                file=os.path.join(root_path, "evaluation_{}_{}_{}.png".format(self.epoch, iteration, i)))
            else:
                points = torch.cumsum(y_hat[i], dim=1).detach().cpu().numpy()
                pixels = np.clip(np.floor(points / local_map_resolution).astype(int) + center, 0,
                                 local_map.shape[0] - 1)
                self._write_png(local_map=local_map, center=center, paths=pixels, others=scan_pixels,
                                file=os.path.join(root_path, "evaluation_{}_{}_{}.png".format(self.epoch, iteration, i)))

    def inference_epoch(self):
        self.set_eval_mode()
        summary_board = SummaryBoard()
        for iteration, data_dict in enumerate(self.val_loader):
            start_time = time.time()

            output_dict, result_dict = self.step(data_dict)
            if iteration % 20 == 0:
                self._display_output(output_dict=output_dict, data_dict=data_dict, iteration=iteration,
                                     root_path="/home/jing/Documents/gn/global_nav/test/evaluation/training/"+self.name)
            th.cuda.synchronize()
            step_time = time.time()
            result_dict = release_cuda(result_dict)
            summary_board.update_from_result_dict(result_dict)
            summary_board.update("step_time", step_time - start_time)

            th.cuda.empty_cache()

        summary_dict = summary_board.summary()
        self.logger.record_dict(summary_dict, prefix="eval/")
        self.logger.dump(step=self.epoch)
        self.set_train_mode()

    def run(self):
        if self.snapshot:
            self.load_snapshot(self.snapshot)

        self.set_train_mode()
        while self.epoch < self.max_epoch:
            self.epoch += 1
            self.run_epoch()
            if self.w_eval > 0:
                self.inference_epoch()


if __name__ == "__main__":
    trainer = Trainer(cfgs=cfg)
    trainer.run_epoch()
    print("test")
