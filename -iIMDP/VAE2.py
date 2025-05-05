import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from scipy.signal import butter, lfilter,find_peaks

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff/fs/2, btype='low')
    y = lfilter(b, a, data)
    return y

def dataload1 ():

    pass
    """The data are processed here and then returned to the network for training, Rs denotes distance, Vs denote SEGF, Vsone is the normalized SEGF"""
    #return Rs,Vs,Vsone


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # 编码器
        self.fc1=nn.LSTM(input_dim, 150, 1, batch_first=True,bidirectional=True)
        self.fc11 = nn.LSTM(300, 100, 1, batch_first=True, bidirectional=True)
        self.fc12 = nn.LSTM(200, 80, 1, batch_first=True, bidirectional=True)
        self.fc21 = nn.Linear(160, latent_dim) # 均值
        self.fc22 = nn.Linear(160, latent_dim) # 对数方差

        # 解码器
        self.fc3 = nn.LSTM(latent_dim, 80, 2, batch_first=True,bidirectional=True)
        self.fc5 = nn.LSTM(160, 200, 2, batch_first=True, bidirectional=True)
        self.fc4 = nn.Linear(400, input_dim)
        self.fc6=nn.Linear(1,80)
        #self.fc9=nn.Linear(80,80)

        #条件映射
        self.fc7 = nn.Linear(80, latent_dim)
        self.fc8 = nn.Linear(latent_dim, latent_dim)
        self.drop=nn.Dropout(0.5)
        self.act=nn.ReLU()
        self.norm=nn.BatchNorm1d(160)
        self.norm2 = nn.BatchNorm1d(30)

        #幅值
        self.fc91=nn.Linear(1,50)
        self.fc92 = nn.Linear(50, 100)
        self.fc93 = nn.Linear(100, 100)
        self.fc94 = nn.Linear(100, 1)
        self.act2=nn.ReLU()


    def amplitude(self,x):
        x=self.act2(self.fc91(x))
        x = self.act2(self.fc92(x))
        x = self.act2(self.fc93(x))
        x = self.fc94(x)
        return x

    def location_con(self,y,z):
        p=self.act(self.fc6(y))
        #p=self.act(self.fc9(p))
        p=self.act(self.norm2(self.fc7(p)))
        p = self.fc8(self.drop(p))
        return z+p#+self.siganl_con(w)

    def encode(self, x):
        h1,_ = self.fc1(x)
        h1, _ = self.fc11(h1)
        h1, _ = self.fc12(h1)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):

        h3 ,_= self.fc3(z)
        h3=self.norm(h3)
        h3,_=self.fc5(h3)
        return self.fc4(h3)

    def forward(self, x,y):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        a=self.amplitude(y)
        return self.decode(self.location_con(y,z)), mu, logvar,a
mse = nn.MSELoss()
# loss function
def loss_function(recon_x, x, mu, logvar,a,wave):
    AE=mse(a*recon_x,wave)
    g1_hat = torch.fft.fft(recon_x, dim=1)
    dataset_hat = torch.fft.fft(x, dim=1)
    g1_rea = torch.real(g1_hat)
    g1_img = torch.imag(g1_hat)
    d_rea = torch.real(dataset_hat)
    d_img = torch.imag(dataset_hat)
    BCE=mse(g1_rea, d_rea) + mse(g1_img, d_img)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD+BCE+AE

# model training
def train():
    # 参数设置
    input_dim = 600  #
    latent_dim = 30  # 潜在空间的维度
    batch_size = 627#56
    # 实例化模型和优化器
    device = torch.device('cuda:0')
    model = VAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    location,waves,wavesone= dataload1()
    wavesone = torch.from_numpy(wavesone).float().to(device)
    waves = torch.from_numpy(waves).float().to(device)
    location = torch.from_numpy(location).float().to(device)

    N=0
    while N < 50000:
        for i in range(627 // batch_size):
            model.train()
            waveone=wavesone[i*batch_size:(i+1)*batch_size]
            wave = waves[i * batch_size:(i + 1) * batch_size]
            loc = location[i * batch_size:(i + 1) * batch_size]

            recon_batch, mu, logvar,a = model(waveone,loc)
            loss = loss_function(recon_batch, waveone, mu, logvar,a,wave)
            loss.backward()
            clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
            N+=1
            print(waveone.shape,N,loss)
            if N%1500==0:
                torch.save(model,'model_fftloss'+str(N)+'R4'+'.pth')

def predictwave(R):
    device = torch.device('cuda:0')
    model=torch.load('model3600.pth')
    model.eval()

    y = torch.from_numpy(R).float().to(device)
    z=torch.randn(np.shape(R)[0], 50).to(device)

    z = model.location_con(y, z)
    sample = model.decode(z)
    A = model.amplitude(y)
    sample *= A
    sample=sample.cpu().detach().numpy()
    return sample


if __name__=="__main__":
    #R=np.array([[24.0],[22.0],[25.0],[30.0]])
    #predictwave(R)
    train()