import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.backbones.diff_hausdorf import HausdorffLoss
from src.configs import ModelType, DataName, LossDictKeys, LossDisType, CollisionLossType, DiversityType
minimum_loss_ratio = 1e-6


class LossEvaluation(nn.Module):
    def __init__(self, cfg):
        super(LossEvaluation, self).__init__()
        self.cfg = cfg
        self.model_type = self.cfg.type
        self.distance_type = self.cfg.distance_type
        self.collision_type = self.cfg.collision_type
        self.scale_waypoints = self.cfg.scale_waypoints
        self.dis_func = HausdorffLoss(mode=self.cfg.hausdorff_dis)

        self.last_point_dis = nn.MSELoss(reduction="mean")

        self.last_ratio = self.cfg.last_ratio
        self.vae_kld_ratio = self.cfg.vae_kld_ratio
        self.dlow_kld_ratio = self.cfg.dlow_kld_ratio
        self.distance_ratio = self.cfg.distance_ratio
        self.diversity_ratio = self.cfg.diversity_ratio
        self.collision_mean_ratio = self.cfg.collision_mean_ratio
        self.collision_max_ratio = self.cfg.collision_max_ratio
        self.coverage_distance_ratio = self.cfg.coverage_distance_ratio
        self.coverage_last_ratio = self.cfg.coverage_last_ratio
        self.coverage_with_last = self.cfg.coverage_with_last
        self.coverage_diverse_ratio = self.cfg.coverage_diverse_ratio
        self.asymmetric_ratio = self.cfg.asymmetric_ratio

        self.sigma = self.cfg.dlow_sigma
        # self.collision_detection_dis = self.cfg.collision_detection_dis
        self.local_map_resolution = self.cfg.local_map_resolution
        self.collision_threshold = self.cfg.collision_threshold
        self.local_map_threshold = self.cfg.local_map_threshold

    def _get_kl_full(self, A, b):
        var = A.bmm(A.transpose(1, 2))
        var_det = torch.det(var)
        var_log = torch.log(var_det)
        KLD = -0.5 * (A.shape[-1] + var_log - b.pow(2).sum(dim=1) - (A * A).sum(dim=-1).sum(dim=-1))
        return KLD.mean()

    def _get_kl_diag(self, A, b):
        var = (A + 0.001) ** 2
        KLD = -0.5 * torch.sum(1 + var.log() - b.pow(2) - var)
        return KLD.mean()

    def _cvae_reconstruction_loss(self, yhat, ygt, y_last):
        distance_loss = []
        for i in range(len(ygt)):
            single_y_gt = ygt[i]
            distance_loss.append(self.dis_func(yhat[i], single_y_gt))
        last_point_loss = self.last_point_dis(yhat[:, -1, :], y_last)
        return torch.stack(distance_loss).mean(), last_point_loss

    def _dlow_reconstruction_loss(self, yhat, ygt, y_last):
        B, N, K, D = yhat.shape
        dtw_losses = []
        single_positions = []
        for i in range(len(ygt)):
            single_y_gt = ygt[i].unsqueeze(0).repeat(N, 1, 1)
            distances = self.dis_func(yhat[i], single_y_gt)
            idx = torch.argmin(distances)
            dtw_losses.append(distances[idx])
            single_positions.append(yhat[i][idx][-1, :])
        final_poses = torch.stack(single_positions)
        last_point_loss = self.last_point_dis(final_poses, y_last)

        distance_loss = torch.stack(dtw_losses).mean()
        # except:
        #     print("error")
        return distance_loss, last_point_loss

    def _dlow_diversity_loss(self, yhat):
        B, N, K, D = yhat.shape
        compare_a = []
        compare_b = []
        for i in range(B):
            for x in range(N):
                for y in range(x + 1, N, 1):
                    if x == y:
                        continue
                    else:
                        compare_a.append(yhat[i][x])
                        compare_b.append(yhat[i][y])
        distance = self.dis_func(torch.stack(compare_a, dim=0), torch.stack(compare_b, dim=0))
        diversity_loss = torch.exp(-distance.mean() / self.sigma)
        return diversity_loss

    def _asymmetric_loss(self, yhat):
        B, N, K, D = yhat.shape
        compare_a = []
        compare_b = []
        for i in range(B):
            for x in range(N):
                for y in range(x + 1, N, 1):
                    if x == y:
                        continue
                    else:
                        compare_a.append(yhat[i][x])
                        compare_b.append(yhat[i][y])
        distance = self.dis_func(torch.stack(compare_a, dim=0), torch.stack(compare_b, dim=0))
        abs_distance = self.dis_func(torch.abs(torch.stack(compare_a, dim=0)), torch.abs(torch.stack(compare_b, dim=0)))
        asymmetric_loss = torch.exp((distance - abs_distance) / self.sigma) - 1.0
        return asymmetric_loss.mean()

    def closest_distance(self, points, segments):
        """
        @ points: N x 2
        @ segments: M x 2
        """
        N, Cp = segments.shape
        p1 = segments[:-1].view(1, N - 1, Cp)  # 1xNxC
        p2 = segments[1:].view(1, N - 1, Cp)  # 1xNxC
        M, Cs = points.shape
        assert Cs == Cp, "dimension should be the same, but get {}, {}".format(Cs, Cp)
        p = points.view(M, 1, Cs)  # Mx1xC
        v = p2 - p1  # 1xNxC
        w = p - p1  # MxNxC
        c1 = torch.sum(w * v, dim=-1)  # MxN
        c2 = torch.sum(v * v, dim=-1)  # 1xN
        b = torch.clamp(c1 / c2, 0, 1).view(M, N - 1, 1)  # MxN
        pb = p1 + b * v  # MxNxC
        d1 = torch.min(torch.norm(p - p1, dim=-1), dim=0)[0]  # N
        d2 = torch.min(torch.norm(p - p2, dim=-1), dim=0)[0]  # N
        d3 = torch.min(torch.norm(p - pb, dim=-1), dim=0)[0]  # N
        pixel_dis = torch.stack((d1, d2, d3), dim=0)

        d = torch.min(pixel_dis, dim=0)[0] * self.local_map_resolution
        d_min = torch.min(d, dim=0)[0]

        d_path = 1 - torch.clamp(d_min, 0.0001, 0.999)
        torch.cuda.empty_cache()
        loss = torch.arctanh(d_path).mean()
        return loss

    def _global_collision(self, yhat, local_map):
        if len(yhat.shape) == 3:
            B, N, C = yhat.shape
            yhat = yhat.view(B, 1, N, C)
        By, S, N, C = yhat.shape
        Bl, W, H = local_map.shape
        assert Bl == By, "the batch shape {} and {} should be the same".format(By, Bl)
        # assert W == H, "the local map width {} not equals to height {}".format(W, H)
        pixel_yhat = yhat / self.local_map_resolution + self.local_map_threshold
        # pixel_yhat = pixel_yhat.to(torch.int)
        all_losses = []
        for i in range(By):
            map_indices = torch.stack(torch.where(local_map[i] > 0), dim=1)
            # _, map_indices = self._process_local_map(local_map[i])
            paths = pixel_yhat[i]
            for j in range(len(paths)):
                all_losses.append(self.closest_distance(points=map_indices, segments=paths[j]))
        return torch.stack(all_losses)

    def _cropped_distance(self, path, single_map):
        N, Cp = path.shape
        M, Cs = single_map.shape
        assert Cs == Cp, "dimension should be the same, but get {}, {}".format(Cs, Cp)
        single_map = single_map.view(M, 1, Cs).to(torch.float)  # Mx1xC
        path = path.view(1, N, Cs)  # 1xNxC
        d = torch.min(torch.norm(single_map - path, dim=-1), dim=0)[0] * self.local_map_resolution  # N

        d_path = 1 - torch.clamp(d, 0.0001, 0.999)
        torch.cuda.empty_cache()
        loss = torch.arctanh(d_path).mean()
        return loss

    def _local_collision(self, yhat, local_map):
        if len(yhat.shape) == 3:
            B, N, C = yhat.shape
            yhat = yhat.view(B, 1, N, C)
        By, S, N, C = yhat.shape
        Bl, W, H = local_map.shape
        assert Bl == By, "the batch shape {} and {} should be the same".format(By, Bl)
        assert W == H, "the local map width {} not equals to height {}".format(W, H)
        pixel_yhat = yhat / self.local_map_resolution + self.local_map_threshold
        pixel_yhat = pixel_yhat.to(torch.int)
        all_losses = []
        for i in range(By):
            map_indices = torch.stack(torch.where(local_map[i] > 0), dim=1)
            paths = pixel_yhat[i]
            for path in paths:
                all_losses.append(self._cropped_distance(path, map_indices))
        return torch.stack(all_losses)

    def _collision_loss(self, yhat, local_map):
        if self.collision_type == CollisionLossType.global_dis:
            return self._global_collision(yhat=yhat, local_map=local_map)
        elif self.collision_type == CollisionLossType.local_dis:
            return self._local_collision(yhat=yhat, local_map=local_map)
        else:
            raise Exception("the collision distance is not defined")

    def process_coverage(self, yhat, y_gts):
        B, N, K, D = yhat.shape
        coverage_last_loss = []
        coverage_distance_loss = []
        if self.coverage_diverse_ratio > minimum_loss_ratio:
            coverage_diverse_loss = []
        for b in range(len(y_gts)):
            if self.coverage_diverse_ratio > minimum_loss_ratio:
                all_indices = set(np.arange(N).tolist())
                selected_indices = set([])

            single_map = y_gts[b]
            all_distances = []
            if self.coverage_with_last:
                last_positions = []
            for i in range(len(single_map)):
                single_y_gt = single_map[i].unsqueeze(0).repeat(N, 1, 1)  # N, K2, D
                distances = self.dis_func(yhat[b], single_y_gt)
                idx = torch.argmin(distances)
                if self.coverage_diverse_ratio > minimum_loss_ratio:
                    selected_indices.add(idx.detach().cpu().item())

                if self.coverage_with_last:
                    last_positions.append(torch.norm(single_y_gt[idx][-1] - yhat[b][idx][-1]))
                all_distances.append(distances[idx])

            if self.coverage_diverse_ratio > minimum_loss_ratio and b == len(y_gts) - 1:
                not_selected_indices = all_indices - selected_indices
                M = len(not_selected_indices)
                N = len(selected_indices)
                if M == 0:
                    pass
                elif N == 0:
                    raise Exception("ground truth is not correct")
                else:
                    not_selected_yh = torch.repeat_interleave(yhat[b][list(not_selected_indices)], N, dim=0)   # M*N, 16, 2
                    selected_yh = yhat[b][list(selected_indices)].repeat((M, 1, 1))  # N*M, 16, 2
                    result = self.dis_func(not_selected_yh, selected_yh).view(M, N)
                    minimum_values = result.min(dim=1)[0]
                    coverage_diverse_loss.append(minimum_values)
            if self.coverage_with_last:
                coverage_last_loss.append(torch.stack(last_positions).mean() + torch.stack(last_positions).max())
            coverage_distance_loss.append(torch.stack(all_distances).mean() + torch.stack(all_distances).max())
        if self.coverage_diverse_ratio > minimum_loss_ratio and len(coverage_diverse_loss) > 0:
            return torch.stack(coverage_distance_loss).mean(), torch.stack(coverage_last_loss).mean(), \
                    torch.exp(torch.stack(coverage_diverse_loss)).mean()
        else:
            return torch.stack(coverage_distance_loss).mean(), torch.stack(coverage_last_loss).mean(), 0

    def _dlow_loss(self, A, b, ygt, yhat, y_last, local_map):
        distance_loss, last_point_loss = self._dlow_reconstruction_loss(yhat=yhat, ygt=ygt, y_last=y_last)
        kld_loss = self._get_kl_diag(A=A, b=b)
        diversity_loss = self._dlow_diversity_loss(yhat=yhat)
        collision_loss = self._collision_loss(yhat=yhat, local_map=local_map)
        loss = self.dlow_kld_ratio * kld_loss + self.diversity_ratio * diversity_loss + \
               self.distance_ratio * distance_loss + self.last_ratio * last_point_loss + \
               self.collision_mean_ratio * collision_loss.mean() + self.collision_max_ratio * collision_loss.max()
        return {LossDictKeys.loss: loss,
                LossDictKeys.last_point_loss: last_point_loss,
                LossDictKeys.diversity_loss: diversity_loss,
                LossDictKeys.dlow_kld_loss: kld_loss,
                LossDictKeys.distance_loss: distance_loss,
                LossDictKeys.collision_loss_mean: collision_loss.mean(),
                LossDictKeys.collision_loss_max: collision_loss.max(), }

    def _cvae_loss(self, mu, logvar, ygt, yhat, y_last, local_map):
        distance_loss, last_point_loss = self._cvae_reconstruction_loss(yhat=yhat, ygt=ygt, y_last=y_last)
        collision_loss = self._collision_loss(yhat=yhat, local_map=local_map)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y_last.shape[0]
        loss = self.last_ratio * last_point_loss + self.vae_kld_ratio * kld_loss + \
               self.distance_ratio * distance_loss + \
               self.collision_mean_ratio * collision_loss.mean() + self.collision_max_ratio * collision_loss.max()
        return {LossDictKeys.loss: loss,
                LossDictKeys.last_point_loss: last_point_loss,
                LossDictKeys.vae_kld_loss: kld_loss,
                LossDictKeys.distance_loss: distance_loss,
                LossDictKeys.collision_loss_mean: collision_loss.mean(),
                LossDictKeys.collision_loss_max: collision_loss.max(), }

    def _dlowae_loss(self, A, b, mu, logvar, ygt, yhat, y_last, y_gts, local_map):
        dlow_kld_loss = self._get_kl_diag(A=A, b=b)
        ae_kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y_last.shape[0]

        loss = self.vae_kld_ratio * ae_kld_loss + self.dlow_kld_ratio * dlow_kld_loss
        output_dict = {LossDictKeys.dlow_kld_loss: self.dlow_kld_ratio * dlow_kld_loss,
                       LossDictKeys.vae_kld_loss: self.vae_kld_ratio * ae_kld_loss, }

        if self.coverage_last_ratio > minimum_loss_ratio and y_gts is not None:
            coverage_path, coverage_last, coverage_diverse = self.process_coverage(yhat=yhat, y_gts=y_gts)
            if coverage_diverse > 0:
                loss += self.coverage_last_ratio * coverage_last + self.coverage_distance_ratio * coverage_path + \
                        self.coverage_diverse_ratio * coverage_diverse
                output_dict.update({LossDictKeys.coverage_distance: self.coverage_distance_ratio * coverage_path,
                                    LossDictKeys.coverage_last: self.coverage_last_ratio * coverage_last,
                                    LossDictKeys.coverage_diverse: self.coverage_diverse_ratio * coverage_diverse, })
            else:
                loss += self.coverage_last_ratio * coverage_last + self.coverage_distance_ratio * coverage_path
                output_dict.update({LossDictKeys.coverage_distance: self.coverage_distance_ratio * coverage_path,
                                    LossDictKeys.coverage_last: self.coverage_last_ratio * coverage_last})
        if self.diversity_ratio > minimum_loss_ratio:
            diversity_loss = self._dlow_diversity_loss(yhat=yhat)
            loss += self.diversity_ratio * diversity_loss
            output_dict.update({LossDictKeys.diversity_loss: self.diversity_ratio * diversity_loss})
        if self.collision_mean_ratio > minimum_loss_ratio or self.collision_max_ratio > minimum_loss_ratio:
            collision_loss = self._collision_loss(yhat=yhat, local_map=local_map)
            loss += self.collision_mean_ratio * collision_loss.mean() + self.collision_max_ratio * collision_loss.max()
            output_dict.update({LossDictKeys.collision_loss_mean: self.collision_mean_ratio * collision_loss.mean(),
                                LossDictKeys.collision_loss_max: self.collision_max_ratio * collision_loss.max()})
        if self.distance_ratio > minimum_loss_ratio:
            distance_loss, last_point_loss = self._dlow_reconstruction_loss(yhat=yhat, ygt=ygt, y_last=y_last)
            loss += self.last_ratio * last_point_loss + self.distance_ratio * distance_loss
            output_dict.update({LossDictKeys.last_point_loss: self.last_ratio * last_point_loss,
                                LossDictKeys.distance_loss: self.distance_ratio * distance_loss, })
        if self.asymmetric_ratio > minimum_loss_ratio:
            asymmetric_loss = self.asymmetric_ratio * self._asymmetric_loss(yhat=yhat)
            loss += asymmetric_loss
            output_dict.update({LossDictKeys.asymmetric_loss: asymmetric_loss})
        output_dict.update({LossDictKeys.loss: loss})
        return output_dict

    def forward(self, output_dict):
        y_gt = output_dict[DataName.path]  # B x K x 2
        y_last = output_dict[DataName.last_poses]
        assert len(y_last.shape) == 2, "y_last shape is not correct"
        y_hat = output_dict[DataName.y_hat]  # B x N x 2
        assert len(y_hat.shape) == 3 or len(y_hat.shape) == 4, "y_hat shape is not correct"
        local_map = output_dict[DataName.png]  # B x W x W
        if self.model_type == ModelType.cvae:
            y_hat_poses = torch.cumsum(y_hat, dim=1) * self.scale_waypoints
            logvar = output_dict[DataName.logvar]  # B x 512
            mu = output_dict[DataName.mu]  # B x 512
            loss_dict = self._cvae_loss(mu=mu, logvar=logvar, yhat=y_hat_poses, ygt=y_gt, y_last=y_last,
                                        local_map=local_map)
        elif self.model_type == ModelType.dlowae:
            if DataName.all_paths in output_dict.keys():
                y_gts = output_dict[DataName.all_paths]  # B x K x 2
            else:
                y_gts = None
            y_hat_poses = torch.cumsum(y_hat, dim=2) * self.scale_waypoints
            A = output_dict[DataName.A]  # B x 512 x 512
            b = output_dict[DataName.b]  # B x 512
            logvar = output_dict[DataName.logvar]  # B x 512
            mu = output_dict[DataName.mu]  # B x 512
            loss_dict = self._dlowae_loss(mu=mu, logvar=logvar, A=A, b=b, yhat=y_hat_poses, ygt=y_gt, y_last=y_last,
                                          y_gts=y_gts, local_map=local_map)
        else:
            raise Exception("model type is not defined")
        return loss_dict
