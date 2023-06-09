import torch
from torch import nn
from src.backbones.vae import CVAE
from src.backbones.transformer import CoverageTransformer


class DiversityDiagAE(CVAE):
    def __init__(self, cfg, activation_func=nn.Softsign):
        super(DiversityDiagAE, self).__init__(cfg, file=cfg.cvae_file, activation_func=activation_func)
        self.paths_num = cfg.paths_num
        self.w_others = cfg.w_others
        if activation_func is None:
            self.mlp = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.2),
                                     nn.Linear(256, 256), nn.LeakyReLU(0.2))
        else:
            self.mlp = nn.Sequential(nn.Linear(512, 256), activation_func(),
                                     nn.Linear(256, 256), activation_func())
        self.head_A = nn.Linear(256, self.zd * self.paths_num)
        self.head_b = nn.Linear(256, self.zd * self.paths_num)

        if self.w_others:
            self.coverage_transformer = CoverageTransformer(d_model=self.zd, num_heads=cfg.transformer_heads)
            if activation_func is None:
                self.coverage_mlp = nn.Sequential(nn.Linear(512 * 2, 512), nn.LeakyReLU(0.2),
                                                  nn.Linear(512, 512), nn.LeakyReLU(0.2))
            else:
                self.coverage_mlp = nn.Sequential(nn.Linear(512 * 2, 512), nn.LeakyReLU(0.2),
                                                  nn.Linear(512, 512), activation_func())

        if cfg.fix_cvae:
            self.set_parameters_fixed()
        else:
            pass

    def forward(self, x):
        mu, logvar, h_x = self.encode(x)
        B, C = h_x.size()  # B x 512
        zs = [self._reparameterize(mu, logvar) for i in range(self.paths_num)]  # P x B x 512
        z = torch.stack(zs, dim=1)
        z = z.view(-1, self.zd)  # B*P x 512

        h = self.mlp(h_x)  # B x 256
        a = self.head_A(h).view(-1, self.zd)  # B*P x 512
        b = self.head_b(h).view(-1, self.zd)  # B*P x 512
        z_hat = a * z + b  # B*P x 512

        h_x_r = h_x.repeat_interleave(self.paths_num, dim=0)  # B*P x 512
        scores = None
        if self.w_others:
            z_hat_others = z_hat.view(B, self.paths_num, self.zd)  # B x P x 512
            others, scores = self.coverage_transformer(z_hat_others)  # B x P x 512
            h_x_r = self.coverage_mlp(torch.concat((others.view(-1, self.zd), h_x_r), dim=-1))

        y_hat = self.decoder(h_x_r, z_hat).view(B, self.paths_num, self.waypoints_num, self.waypoint_dim)
        return y_hat, a, b, mu, logvar, scores

    def deterministic_forward(self, x):
        mu, logvar, h_x = self.encode(x)
        B, C = h_x.size()  # B x 512
        zs = [mu for i in range(self.paths_num)]
        z = torch.stack(zs, dim=1).view(-1, self.zd)

        h = self.mlp(h_x)
        a = self.head_A(h).view(-1, self.zd)
        b = self.head_b(h).view(-1, self.zd)
        z_hat = a * z + b

        h_x_r = h_x.repeat_interleave(self.paths_num, dim=0)
        scores = None
        if self.w_others:
            z_hat_others = z_hat.view(B, self.paths_num, self.zd)  # B x P x 512
            others, scores = self.coverage_transformer(z_hat_others)  # B x P x 512
            h_x_r = self.coverage_mlp(torch.concat((others.view(-1, self.zd), h_x_r), dim=-1))

        y_hat = self.decoder(h_x_r, z_hat).view(B, self.paths_num, self.waypoints_num, self.waypoint_dim)
        return y_hat, a, b, mu, logvar
