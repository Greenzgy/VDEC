import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VDECCNN(nn.Module):
    def __init__(self, input_c=1, h1_c=64, h2_c=64, h3_c=256, h4_c=256, h5_c=1024, z_dim=10, k=4, s=2, p=1, class_num=10):
        super(VDECCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_c, h1_c, k, s, p),  # B, 64, 16, 16
            nn.ReLU(True),
            nn.Conv2d(h1_c, h2_c, k, s, p),  # B, 64, 8, 8
            nn.ReLU(True),
            nn.Conv2d(h2_c, h3_c, k, s, p),  # B, 256, 4, 4
            nn.ReLU(True),
            nn.Conv2d(h3_c, h4_c, k, s, p),  # B, 256, 2, 2
            nn.ReLU(True),
            View((-1, h5_c))  # B, 1024
        )

        self.mean = nn.Sequential(nn.Linear(h5_c, z_dim))
        self.logsigma2 = nn.Sequential(nn.Linear(h5_c, z_dim))
        self.alpha = nn.Sequential(nn.Linear(h5_c, z_dim))

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h5_c),  # B, 1024
            nn.ReLU(True),
            View((-1, h4_c, 2, 2)),  # B, 256, 2, 2
            nn.ConvTranspose2d(h4_c, h3_c, k, s, p),  # B, 256, 4, 4
            nn.ReLU(True),
            nn.ConvTranspose2d(h3_c, h2_c, k, s, p),  # B, 64, 8, 8
            nn.ReLU(True),
            nn.ConvTranspose2d(h2_c, h1_c, k, s, p),  # B, 64, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(h1_c, input_c, k, s, p)  # B, 1, 32, 32
        )

        self.Ex = nn.Parameter(torch.FloatTensor(class_num, z_dim), requires_grad=True)
        self.Enn = nn.Parameter(torch.FloatTensor(class_num, z_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.Ex.data)
        torch.nn.init.xavier_normal_(self.Enn.data)

    def encode(self, x):
        h = self.encoder(x)
        return self.mean(h), self.logsigma2(h), self.alpha(h)

    def reparameterize(self, ex, logen2, he):
        en = torch.exp(logen2 / 2)
        noise1 = torch.randn_like(en)
        noise2 = torch.randn_like(noise1)
        z = ex + noise1 * en + noise1 * noise2 * he
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        ex, logen2, he = self.encode(x)
        z = self.reparameterize(ex, logen2, he)
        x_reconst = self.decode(z)

        eps = 1e-8
        one_term = ex.pow(2)
        two_term = torch.exp(logen2)
        three_term = logen2

        q = torch.exp(
            -(torch.sum(torch.pow(z.unsqueeze(1) - self.Ex.unsqueeze(0), 2) / (2 * torch.pow(self.Enn.unsqueeze(0), 2)),
                        2)))

        q = (q.t() / torch.sum(q, 1)).t()
        q = q + eps

        return x_reconst, one_term, two_term, three_term, q, z


class VDECUSPS(nn.Module):
    def __init__(self, class_num=10, input_c=1, h1_c=64, h2_c=64, h3_c=256, h4_c=1024, z_dim=10, k=4, s=2, p=1):
        super(VDECUSPS, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_c, h1_c, k, s, p),  # B, 64, 8, 8
            nn.ReLU(True),
            nn.Conv2d(h1_c, h2_c, k, s, p),  # B, 64, 4, 4
            nn.ReLU(True),
            nn.Conv2d(h2_c, h3_c, k, s, p),  # B, 256, 2, 2
            nn.ReLU(True),
            View((-1, h4_c))  # B, 1024
        )

        self.mean = nn.Sequential(nn.Linear(h4_c, z_dim))
        self.logsigma2 = nn.Sequential(nn.Linear(h4_c, z_dim))
        self.alpha = nn.Sequential(nn.Linear(h4_c, z_dim))

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h4_c),  # B, 1024
            nn.ReLU(True),
            View((-1, h3_c, 2, 2)),  # B, 256, 2, 2
            nn.ConvTranspose2d(h3_c, h2_c, k, s, p),  # B, 64, 4, 4
            nn.ReLU(True),
            nn.ConvTranspose2d(h2_c, h1_c, k, s, p),  # B, 64, 8, 8
            nn.ReLU(True),
            nn.ConvTranspose2d(h1_c, input_c, k, s, p)  # B, 1, 16, 16
        )

        self.Ex = nn.Parameter(torch.FloatTensor(class_num, z_dim), requires_grad=True)
        self.Enn = nn.Parameter(torch.FloatTensor(class_num, z_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.Ex.data)
        torch.nn.init.xavier_normal_(self.Enn.data)

    def encode(self, x):
        h = self.encoder(x)
        return self.mean(h), self.logsigma2(h), self.alpha(h)

    def reparameterize(self, ex, logen2, he):
        en = torch.exp(logen2 / 2)
        noise1 = torch.randn_like(en)
        noise2 = torch.randn_like(noise1)
        z = ex + noise1 * en + noise1 * noise2 * he
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        ex, logen2, he = self.encode(x)
        z = self.reparameterize(ex, logen2, he)
        x_reconst = self.decode(z)

        eps = 1e-8
        one_term = ex.pow(2)
        two_term = torch.exp(logen2)
        three_term = logen2

        q = torch.exp(
            -(torch.sum(torch.pow(z.unsqueeze(1) - self.Ex.unsqueeze(0), 2) / (2 * torch.pow(self.Enn.unsqueeze(0), 2)),
                        2)))

        q = (q.t() / torch.sum(q, 1)).t()
        q = q + eps

        return x_reconst, one_term, two_term, three_term, q, z


class VDECMlp(nn.Module):
    def __init__(self, class_num, input_dim, h1_dim=500, h2_dim=500, h3_dim=2000, z_dim=10):
        super(VDECMlp, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1_dim),  # B, 500
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),  # B, 500
            nn.ReLU(),
            nn.Linear(h2_dim, h3_dim),  # B, 2000
            nn.ReLU(),
        )

        self.mean = nn.Sequential(nn.Linear(h3_dim, z_dim))
        self.logsigma2 = nn.Sequential(nn.Linear(h3_dim, z_dim))
        self.alpha = nn.Sequential(nn.Linear(h3_dim, z_dim))

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h3_dim),  # B, 2000
            nn.ReLU(),
            nn.Linear(h3_dim, h2_dim),  # B, 500
            nn.ReLU(),
            nn.Linear(h2_dim, h1_dim),  # B, 500
            nn.ReLU(),
            nn.Linear(h1_dim, input_dim)  # B, input_dim
        )

        self.Ex = nn.Parameter(torch.FloatTensor(class_num, z_dim), requires_grad=True)
        self.Enn = nn.Parameter(torch.FloatTensor(class_num, z_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.Ex.data)
        torch.nn.init.xavier_normal_(self.Enn.data)

    def encode(self, x):
        h = self.encoder(x)
        return self.mean(h), self.logsigma2(h), self.alpha(h)

    def reparameterize(self, ex, logen2, he):
        en = torch.exp(logen2 / 2)
        noise1 = torch.randn_like(en)
        noise2 = torch.randn_like(noise1)
        z = ex + noise1 * en + noise1 * noise2 * he
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        ex, logen2, he = self.encode(x)
        z = self.reparameterize(ex, logen2, he)
        x_reconst = self.decode(z)

        eps = 1e-8
        one_term = ex.pow(2)
        two_term = torch.exp(logen2)
        three_term = logen2

        q = torch.exp(
            -(torch.sum(torch.pow(z.unsqueeze(1) - self.Ex.unsqueeze(0), 2) / (2 * torch.pow(self.Enn.unsqueeze(0), 2)),
                        2)))

        q = (q.t() / torch.sum(q, 1)).t()
        q = q + eps

        return x_reconst, one_term, two_term, three_term, q, z
