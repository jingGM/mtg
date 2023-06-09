import torch
from torch import nn
from src.backbones.rnn import RNNDecoder


class CVAE(nn.Module):
    def __init__(self, cfg, file=None, activation_func=nn.Softsign):
        super(CVAE, self).__init__()
        self.zd = cfg.vae_zd
        self.waypoint_dim = cfg.waypoint_dim
        self.waypoints_num = cfg.waypoints_num
        self.output_dim = self.waypoint_dim * self.waypoints_num
        # baseline
        # self.baseline = None

        # encode
        if activation_func is None:
            self.encoder = nn.Sequential(nn.Linear(cfg.perception_in, 1024), nn.LeakyReLU(0.1),
                                         nn.Linear(1024, 2048), nn.LeakyReLU(0.2),
                                         nn.Linear(2048, 512), nn.LeakyReLU(0.2),
                                         nn.Linear(512, 512), nn.LeakyReLU(0.2))
        else:
            self.encoder = nn.Sequential(nn.Linear(cfg.perception_in, 1024), activation_func(),
                                         nn.Linear(1024, 2048), activation_func(),
                                         nn.Linear(2048, 512), activation_func(),
                                         nn.Linear(512, 512), activation_func())
        self.e_mu = nn.Linear(512, self.zd)
        self.e_logvar = nn.Linear(512, self.zd)

        # decode
        self.decoder = RNNDecoder(in_dim=self.zd * 2, out_dim=self.waypoint_dim, hidden_dim=self.zd,
                                  steps=self.waypoints_num, rnn_type=cfg.rnn_type,
                                  output_threshold=cfg.vae_output_threshold, activation_func=activation_func)

        if file is not None:
            self._load_states(file)

    def set_parameters_fixed(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.e_mu.parameters():
            param.requires_grad = False
        for param in self.e_logvar.parameters():
            param.requires_grad = False

    def _load_states(self, file):
        state_dict = torch.load(file, map_location=torch.device('cpu'))
        model_dict = state_dict['model']
        self.load_state_dict(model_dict, strict=False)

    def encode(self, observation):
        h = self.encoder(observation)  # B x 512
        # if self.baseline is not None:
        #     y_hat = self.baseline(observation)  # B x Hat_dim
        return self.e_mu(h), self.e_logvar(h), h    # B x zd

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z):
        return self.decoder(x_pre=x, z=z)

    def deterministic_forward(self, observation):
        mu, logvar, xh = self.encode(observation)
        return self.decode(xh, mu), mu, logvar

    def sample_forward(self, observation, N):
        mu, logvar, xh = self.encode(observation)
        B, C = mu.shape
        sampled_z = [self._reparameterize(mu, logvar) for i in range(N)]
        z = torch.stack(sampled_z).view(-1, C)
        h_x_r = xh.repeat_interleave(N, dim=0)
        y_hat = torch.transpose(self.decode(h_x_r, z).view(N, B, self.waypoints_num, self.waypoint_dim), dim0=0, dim1=1)
        return y_hat, mu, logvar

    def forward(self, observation):
        mu, logvar, xh = self.encode(observation)
        z = self._reparameterize(mu, logvar)
        return self.decode(xh, z), mu, logvar