import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.functional import pad

from utility import models, sdr
from conv_stft import STFT
from deepbeam import R_global, dataset

torch.pi = torch.acos(torch.zeros(1)).item() * 2


# Conv-TasNet
class TasNet(nn.Module):
    def __init__(self, enc_dim=97, feature_dim=128, sr=48000, win=2.5, layer=8, stack=3,
                 kernel=3, num_spk=1, causal=True):
        super(TasNet, self).__init__()
        
        # hyper parameters
        self.num_spk = num_spk

        self.pairs = ((0, 3), (1, 4), (2, 5), (0, 1), (2, 3), (4, 5))
        self.ori_pairs = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5))

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.sr = sr
        self.win = int(self.sr*win/1000)
        self.stride = self.win // 2
        self.frequency_vector = np.arange(enc_dim) * self.sr / enc_dim
        self.layer = layer
        self.stack = stack
        self.kernel = kernel
        self.n_grid = 36
        self.causal = causal
        self.channel = 1
        self.n_mic = len(self.pairs)
        self.mic_array_layout = R_global-np.tile(R_global[:,0].reshape((3,1)), (1, self.n_mic))
        frame = 602
        z = np.zeros((self.n_mic, self.n_grid, frame))
        a = np.arange(frame).reshape(1, 1, frame)
        m_data = torch.from_numpy(z + a)

        delay = np.zeros((self.n_mic, self.n_grid))
        for h, m in enumerate(self.ori_pairs):
            dx = self.mic_array_layout[0, m[1]] - self.mic_array_layout[0, m[0]]
            dy = self.mic_array_layout[1, m[1]] - self.mic_array_layout[1, m[0]]
            for i in range(self.n_grid):
                delay[h, i] = dx * np.cos(i * 10) + dy * np.sin(i * 10)
        delay = torch.from_numpy(delay).unsqueeze(dim=-1).expand(-1, -1, frame)
        self.w = torch.exp(-2j * np.pi * m_data * delay)/343
        # input encoder

        self.encoder = nn.Conv1d(self.channel, self.enc_dim, self.win, bias=False, stride=self.stride)
        
        # TCN separator
        self.TCN_DIM = 22213
        self.TCN = models.TCN(self.TCN_DIM, self.enc_dim*self.num_spk, self.feature_dim, self.feature_dim*4,
                              self.layer, self.stack, self.kernel, causal=self.causal)

        self.receptive_field = self.TCN.receptive_field
        
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)
        filter_length = int(self.sr * 0.004)
        hop_length = int(
            self.sr * 0.00125)  # doesn't need to be specified. if not specified, it's the same as filter_length
        window = 'hann'
        win_length = int(self.sr * 0.0025)
        self.stft = STFT(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            window=window
        )

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, self.n_mic, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, self.n_mic, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)
        self.u1 = 0
        self.u2 = 0

        return input, rest
        
    def forward(self, input, angle):
        
        # padding
        mag, ph = self.stft.transform(input.reshape(-1, input.size()[-1]))
        pad = Variable(torch.zeros(mag.size()[0],mag.size()[1], 1)).type(input.type())
        mag = torch.cat([mag, pad], -1)
        ph = torch.cat([ph, pad], -1)
        output, rest = self.pad_signal(input)
        enc_output = self.encoder(output[:, :1])  # B, N, L
        mag = mag.view(enc_output.size(0), self.n_mic, -1, enc_output.size(-1))
        ph = ph.view(enc_output.size(0), self.n_mic, -1, enc_output.size(-1))
        LPS = 10 * torch.log10(mag ** 2 + 10e-20)

        complex = (mag * torch.exp(ph * 1j))
        IPD_list = []
        for m in self.pairs:
            com_u1 = complex[:, m[0]]
            com_u2 = complex[:, m[1]]
            IPD = torch.angle(com_u1 * torch.conj(com_u2))
            IPD /= (self.frequency_vector + 1.0)[:, None]
            IPD = IPD % torch.pi
            IPD = IPD.unsqueeze(dim=1)
            IPD_list.append(IPD)
        IPD = torch.cat(IPD_list, dim=1)
        steering_vector = self.__get_steering_vector(angle, self.pairs)
        steering_vector = steering_vector.unsqueeze(dim=-1)
        AF = steering_vector * IPD
        AF = AF/AF.sum(dim=1, keepdims=True).real
        w = self.w.unsqueeze(dim=0).expand(AF.size()[0], -1, -1, -1)
        w = w.unsqueeze(dim=3).expand(-1, -1, -1, self.enc_dim, -1)
        w_mag = w * torch.conj(w)
        y_mag = mag.unsqueeze(dim=2).expand(-1, -1, self.n_grid, -1, -1) ** 2
        dpr = (w_mag * y_mag)/((w_mag * y_mag).sum(2, keepdims=True)).real
        dpr = torch.unbind(dpr, dim=2)
        feature_list = [enc_output.unsqueeze(dim=1), AF, torch.cos(IPD)]
        feature_list.extend(dpr)
        fusion = torch.cat(feature_list, dim=1).float()

        batch_size = output.size(0)
        fusion = fusion.view(batch_size, -1, fusion.size()[-1])
        
        # waveform encoder


        masks = torch.sigmoid(self.TCN(fusion)).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L
        masked_output = enc_output.unsqueeze(1) * masks  # B, C, N, L
        
        # waveform decoder
        output = self.decoder(masked_output.view(batch_size*self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        output = output.view(batch_size, self.num_spk, -1)  # B, C, T

        return output


    def __get_steering_vector(self, angle, pairs):
        steering_vector = np.zeros((len(angle), len(self.frequency_vector), 6), dtype='complex')

        # get delay
        delay = np.zeros((len(angle), self.n_mic))
        print(self.mic_array_layout.shape)
        for h, m in enumerate(pairs):
            dx = self.mic_array_layout[0, m[1]] - self.mic_array_layout[0, m[0]]
            dy = self.mic_array_layout[1, m[1]] - self.mic_array_layout[1, m[0]]
            delay[:, h] = dx * np.cos(angle) + dy * np.sin(angle)

        for f, frequency in enumerate(self.frequency_vector):
            for m in range(len(self.pairs)):
                steering_vector[:, f, m] = np.exp(1j * 2 * np.pi * frequency * delay[:, m] / 343)
        steering_vector = torch.from_numpy(steering_vector)

        return torch.transpose(steering_vector, 1, 2)




def test_conv_tasnet():
    x = np.expand_dims(dataset.__getitem__(0)[0], axis=0)
    x = torch.from_numpy(x).float()
    print(x.size())
    angle = np.array([2.4])
    nnet = TasNet()
    pytorch_total_params = sum(p.numel() for p in nnet.parameters() if p.requires_grad)
    x = nnet(x, angle)
    print(x.size())

if __name__ == "__main__":
    # test_conv_tasnet()
    #a = __get_steering_vector(0.3, 16000, 33)
    test_conv_tasnet()
    #print(a.shape)


