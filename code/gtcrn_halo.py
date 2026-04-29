"""
GTCRN + HALO
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2-2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1]+1] = 1- erb_filters[-2, bins[-2]:bins[-1]+1]
        
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))
    
    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))
        
    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)  #门控循环神经网络，用于时间建模，输出维度是 channels × 2
        self.att_fc = nn.Linear(channels*2, channels)   #降维到原始通道数
        self.att_act = nn.Sigmoid()    #激活函数

    def forward(self, x):    
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)  对频率维度求平均，得到时间-通道上的平均能量特征 zt
        at = self.att_gru(zt.transpose(1,2))[0]  # (B, T, C) → (B, T, 2C)
        at = self.att_fc(at).transpose(1,2)   # (B, T, C) → (B, C, T)
        at = self.att_act(at)  # sigmoid 激活，变成注意力权重
        At = at[..., None]  # (B,C,T,1)

        return x * At


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConvTime(nn.Module):
    """只沿时间维的深度可分离卷积"""
    def __init__(self, channels, k=5, stride=1, causal=True, lookahead=0):
        super().__init__()
        self.k, self.causal, self.lookahead = k, causal, lookahead
        self.dw = nn.Conv2d(channels, channels, kernel_size=(k,1), stride=(stride,1),
                            padding=(0,0), groups=channels, bias=False)
    def forward(self, x):               # x: (B,C,T,F)
        if self.causal:
            pad_left = self.k - 1 - self.lookahead
            pad_right = self.lookahead
        else:
            pad_left = (self.k - 1) // 2
            pad_right = self.k - 1 - pad_left
        x = F.pad(x, [0, 0, pad_left, pad_right])  # [Wl,Wr,Ht,Hb]; H=时间
        return self.dw(x)

class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
    
        self.sfe = SFE(kernel_size=3, stride=1)
        
        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)
        
        self.tra = TRA(in_channels//2)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B,2C,T,F)
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))

        h1 = self.tra(h1)

        x =  self.shuffle(h1, x2)
        
        return x

class GatedDown2(nn.Module):
    """
    HALO_Restraction
    """
    def __init__(self, in_ch=4, out_ch=2, K=5, gate_hidden=8, gate_act="softmax"):
        super().__init__()
        self.K = K
        self.bases = nn.ModuleList([nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True) for _ in range(K)])
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, gate_hidden, kernel_size=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(gate_hidden, K, kernel_size=1, bias=True),
        )
        assert gate_act in ("softmax", "sigmoid")
        self.gate_act = gate_act

    def forward(self, x_pair):                       # (B, in_ch, L, F)
        gates = self.gate(x_pair)                    # (B, K, L, F)
        if self.gate_act == "softmax":
            gates = F.softmax(gates, dim=1)          
        else:
            gates = torch.sigmoid(gates)
            gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)
        y = 0
        for k, base in enumerate(self.bases):        
            y = y + base(x_pair) * gates[:, k:k+1, :, :]   
        return y                                     # (B, out_ch=2, L, F)


class GatedUp2(nn.Module):
    """
    HALO_Restoration
    """
    def __init__(self, in_ch=2, out_ch=2, K=2, gate_hidden=8):
        super().__init__()
        self.K = K
        self.bases_even = nn.ModuleList([nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True) for _ in range(K)])
        self.bases_odd  = nn.ModuleList([nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True) for _ in range(K)])
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, gate_hidden, kernel_size=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(gate_hidden, 2*K, kernel_size=1, bias=True),  
        )

    def forward(self, z, T_ref: int | None = None):  # z: (B,2,L,F)
        B, C, L, Freq = z.shape
        g = self.gate(z)                              # (B, 2K, L, F)
        g = g.view(B, 2, self.K, L, Freq)            # (B, 2, K, L, F)
        g_even = F.softmax(g[:, 0], dim=1)           # (B, K, L, F)
        g_odd  = F.softmax(g[:, 1], dim=1)           # (B, K, L, F)

        ye = 0
        for k, be in enumerate(self.bases_even):
            ye = ye + be(z) * g_even[:, k:k+1, :, :] # (B,2,L,F)
        yo = 0
        for k, bo in enumerate(self.bases_odd):
            yo = yo + bo(z) * g_odd[:, k:k+1, :, :]  # (B,2,L,F)

        y = torch.zeros(B, C, 2*L, Freq, device=z.device, dtype=z.dtype)
        y[:, :, 0::2, :] = ye
        y[:, :, 1::2, :] = yo
        if T_ref is not None and y.size(2) > T_ref:
            y = y[:, :, :T_ref, :]
        return y                                      # (B,2,2L or T_ref,F)


class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size)
        """
        if h== None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h
    
    
class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)
    
    def forward(self, x):
        """x: (B, C, T, F)"""
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size) # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0,2,1,3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) 
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
        return dual_out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, 20, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(20, 24, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            GTConvBlock(24, 24, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            GTConvBlock(24, 24, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            GTConvBlock(24, 24, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(24, 24, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), use_deconv=True),
            GTConvBlock(24, 24, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True),
            GTConvBlock(24, 24, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True),
            ConvBlock(24, 20, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(20, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers-1-i])
        return x
    

class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s
      

class GTCRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)

        self.encoder = Encoder()
        
        self.dpgrnn1 = DPGRNN(24, 33, 24)
        self.dpgrnn2 = DPGRNN(24, 33, 24)
        
        self.decoder = Decoder()

        self.mask = Mask()
        self.down2 = GatedDown2(in_ch=4, out_ch=2, K=5, gate_hidden=8, gate_act="softmax")
        self.up2   = GatedUp2(in_ch=2, out_ch=2, K=5, gate_hidden=8)  


    def forward(self, spec):
        """
        spec: (B, F, T, 2)
        """
        spec_ref = spec  # (B,F,T_full,2)
        B, F, T_full, _ = spec.shape
        spec = spec.permute(0,3,2,1)

        cur_full = spec[:, :, 0::2, :]                     # (B, 2, ceil(T_full/2), F)
        prev_half = spec[:, :, 1::2, :]                    # (B, 2, floor(T_full/2), F)
        if prev_half.size(2) < cur_full.size(2):
            prev_pad = torch.zeros_like(cur_full[:, :, :1, :])   # (B,2,1,F) 
            prev_half = torch.cat([prev_pad, prev_half], dim=2)  # (B,2,ceil(T_full/2),F)
        x_pair = torch.cat([prev_half, cur_full], dim=1)         # (B,4,ceil(T_full/2),F)
        spec = self.down2(x_pair)                                # (B,2,ceil(T_full/2),F)

        spec_real = spec[:, 0, :, :]
        spec_imag = spec[:, 1, :, :]
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)

        feat = self.erb.bm(feat)  # (B,3,T,129)
        feat = self.sfe(feat)     # (B,9,T,129)

        feat, en_outs = self.encoder(feat)
        
        feat = self.dpgrnn1(feat) # (B,16,T,33)
        feat = self.dpgrnn2(feat) # (B,16,T,33)

        m_feat = self.decoder(feat, en_outs)
        
        m = self.erb.bs(m_feat)

        out = self.up2(m, T_ref=T_full)               # (B, 4, T, F)

        spec_enh = self.mask(out, spec_ref.permute(0,3,2,1)) # (B,2,T_full,F)
        spec_enh = spec_enh.permute(0, 3, 2, 1)         # (B, F, T_full, 2)
        return spec_enh



if __name__ == "__main__":
    model = GTCRN().eval()

    """complexity count"""
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (257, 126, 2), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print(flops, params)
    
    """causality check"""
    a = torch.randn(1, 16000)
    b = torch.randn(1, 16000)
    c = torch.randn(1, 16000)
    x1 = torch.cat([a, b], dim=1)
    x2 = torch.cat([a, c], dim=1)
    
    x1 = torch.stft(x1, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=True)
    x2 = torch.stft(x2, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=True)
    y1 = model(torch.view_as_real(x1))[0].contiguous()
    y2 = model(torch.view_as_real(x2))[0].contiguous()
    y1 = torch.istft(torch.view_as_complex(y1), 512, 256, 512, torch.hann_window(512).pow(0.5))
    y2 = torch.istft(torch.view_as_complex(y2), 512, 256, 512, torch.hann_window(512).pow(0.5))
    
    print((y1[:16000-256*2] - y2[:16000-256*2]).abs().max())
    print((y1[16000:] - y2[16000:]).abs().max())
