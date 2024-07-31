import torch
import torch.nn as nn
from .SPE import SPE
from .blocks import SelfAttention1d, CrossAttention1d
from einops import rearrange, reduce, repeat
import gin
from .Encoders import Encoder1D

@gin.configurable
class ConvBlock1D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 skip_channels,
                 time_cond_channels,
                 time_channels,
                 cond_channels,
                 kernel_size,
                 act=nn.SiLU,
                 res=True):
        super().__init__()
        self.res = res

        self.conv1 = nn.Conv1d(in_c + skip_channels + time_cond_channels,
                               out_c,
                               kernel_size=kernel_size,
                               padding="same")
        self.gn1 = nn.GroupNorm(min(16, out_c), out_c)
        self.conv2 = nn.Conv1d(out_c,
                               out_c,
                               kernel_size=kernel_size,
                               padding="same")
        self.gn2 = nn.GroupNorm(min(16, out_c), out_c)
        self.act = act()

        self.time_mlp = nn.Sequential(nn.Linear(time_channels, 128), act(),
                                      nn.Linear(128, 2 * out_c))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_channels, 128), act(),
                                      nn.Linear(128, 2 * out_c))

        if skip_channels:
            self.to_out = nn.Conv1d(in_c,
                                    out_c,
                                    kernel_size=1,
                                    padding="same")
        else:
            self.to_out = nn.Identity()
            
        self.time_shape = out_c
        self.cond_shape = out_c

    def forward(self, x, time=None, skip=None, zsem=None, time_cond=None):
        res = x.clone()


        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        if time_cond is not None:
            x = torch.cat([x, time_cond], dim=1)

        x = self.conv1(x)

        x = self.gn1(x)
        x = self.act(x) 
        
        
        time = self.time_mlp(time)
        time_mult, time_add = torch.split(time, time.shape[-1] // 2, -1)
        
        x = x * time_mult[:, :, None] + time_add[:, :, None]

        
        if zsem is not None:
            zsem = self.cond_mlp(zsem)
            zsem_mult, zsem_add = torch.split(zsem, zsem.shape[-1] // 2, -1)

            x = x * zsem_mult[:, :, None] + zsem_add[:, :, None]
         

        x = self.act(x)

        x = self.conv2(x)
        x = self.gn2(x)

        x = self.act(x)

        if self.res:
            return x + self.to_out(res)

        return x

@gin.configurable
class EncoderBlock1D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 time_cond_channels,
                 time_channels,
                 cond_channels,
                 kernel_size=3,
                 ratio=2,
                 act=nn.SiLU,
                 use_self_attn=False,
                 cross_attn_channels = 0):
        super().__init__()
        self.conv = ConvBlock1D(in_c=in_c,
                                out_c=in_c,
                                time_cond_channels=time_cond_channels,
                                skip_channels = 0,
                                time_channels=time_channels,
                                cond_channels=cond_channels,
                                kernel_size=kernel_size,
                                act=act)

        self.self_attn = SelfAttention1d(
            in_c, 4) if use_self_attn else nn.Identity()
        self.cross_attn_channels = cross_attn_channels  
        
        if self.cross_attn_channels:
            print(" ONe Encoder block with cross attention")
            self.cross_attn = CrossAttention1d(
                in_c, cross_attn_channels, 4)

        if ratio == 1:
            self.pool = nn.Conv1d(in_c,
                                  out_c,
                                  kernel_size=kernel_size,
                                  padding="same")
        else:
            self.pool = nn.Conv1d(in_c,
                                  out_c,
                                  kernel_size=kernel_size,
                                  stride=ratio,
                                  padding=(kernel_size) // 2)

    def forward(self, inputs, time, cond=None, time_cond=None, zsem=None , context=None):
        skip = self.conv(inputs, time=time, zsem=zsem, time_cond=time_cond)
        skip = self.self_attn(skip)
        
        if self.cross_attn_channels:
            skip = self.cross_attn(input=skip, context=context)
        x = self.pool(skip)
        return x, skip

@gin.configurable
class MiddleBlock1D(nn.Module):

    def __init__(self,
                 in_c,
                 time_cond_channels,
                 time_channels,
                 cond_channels,
                 kernel_size=3,
                 ratio=2,
                 act=nn.SiLU,
                 use_self_attn=False,
    ):
        super().__init__()
        self.conv = ConvBlock1D(in_c=in_c,
                                out_c=in_c,
                                skip_channels = 0,
                                time_cond_channels=time_cond_channels,
                                time_channels=time_channels,
                                cond_channels=cond_channels,
                                kernel_size=kernel_size,
                                act=act)
        self.self_attn = SelfAttention1d(
            in_c, in_c // 32) if use_self_attn else nn.Identity()

            
    def forward(self, x, time, time_cond=None, zsem=None):
        x = self.conv(x, time=time, zsem=zsem, time_cond=time_cond)
        x = self.self_attn(x)
        return x

@gin.configurable
class DecoderBlock1D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 time_cond_channels,
                 time_channels,
                 cond_channels,
                 kernel_size,
                 act=nn.SiLU,
                 ratio=2,
                 use_self_attn=False,
                 skip_size = None,
                 cross_attn_channels = 0):
        super().__init__()
        if ratio == 1:
            if in_c == out_c:
                self.up = nn.Identity()
            else:
                self.up = nn.Conv1d(in_c,
                                    out_c,
                                    kernel_size=3,
                                    stride=1,
                                    padding="same")
        else:
            #self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2)
            self.up = nn.Sequential(
                nn.Upsample(mode='nearest', scale_factor=ratio),
                nn.Conv1d(in_c, out_c, kernel_size=3, stride=1,
                          padding="same"))

        self.conv = ConvBlock1D(in_c=out_c,
                                out_c=out_c,
                                time_cond_channels=time_cond_channels,
                                skip_channels = skip_size if skip_size is not None else out_c,
                                time_channels=time_channels,
                                cond_channels=cond_channels,
                                kernel_size=kernel_size,
                                act=act)
        self.self_attn = SelfAttention1d(
            out_c, 4) if use_self_attn else nn.Identity()

        self.cross_attn_channels = cross_attn_channels
        if cross_attn_channels:
            print(" ONe decoder block with cross attention")
            self.cross_attn = CrossAttention1d(
                out_c, cross_attn_channels, 4)

    def forward(self, x, skip, time, time_cond=None, zsem=None, context = None):
        x = self.up(x)

        x = self.conv(x, time=time, skip=skip, zsem=zsem, time_cond=time_cond)
        
        x = self.self_attn(x)
        
        if self.cross_attn_channels:
            x = self.cross_attn(x, context=context)
        return x

@gin.configurable
class UNET1D(nn.Module):

    def __init__(self,
                 in_size=128,
                 out_size = None,
                 channels=[128, 128, 256, 256],
                 ratios=[2, 2, 2, 2, 2],
                 kernel_size=5,
                 cond={},
                 time_channels=64,
                 time_cond_in_channels=1,
                 time_cond_channels=64,
                 z_channels=32,
                 n_attn_layers=0,
                 cond_type = "all",
                 past_cross_attention = [0,0,0,0],
                 context_channels = 0):
        
        super().__init__()
        
        self.channels = channels
        self.time_cond_channels = time_cond_channels
        self.time_cond_in_channels = time_cond_in_channels
        self.past_cross_attention = past_cross_attention
        self.context_channels = context_channels
        self.in_size = in_size
        if out_size is None:
            out_size = in_size

        n = len(self.channels)
        ratios = [1] + ratios   
        
        if time_channels == 0:
            self.time_emb = lambda _: torch.empty(0)
        else:
            self.time_emb = SPE(time_channels)

        self.cond_modules = nn.ModuleDict()
        self.cond_keys = list(cond.keys())

        c_channels = 0
        for key, p in cond.items():
            if p["num_classes"] == 0:
                self.cond_modules[key] = nn.Sequential(
                    nn.Linear(1, p["emb_dim"]), nn.ReLU())

            else:
                self.cond_modules[key] = nn.Embedding(p["num_classes"] + 1,
                                                      p["emb_dim"])
            c_channels += p["emb_dim"]

        cond_channels = c_channels + z_channels

        if time_cond_channels and cond_type=="all":
            self.cond_emb_time = nn.ModuleList()
            self.cond_emb_time.append(
                nn.Sequential(
                    nn.Conv1d(time_cond_in_channels,
                              time_cond_channels,
                              kernel_size=kernel_size,
                              stride=1,
                              padding="same"), nn.SiLU()))
            for i in range(0, n):
                self.cond_emb_time.append(
                    nn.Sequential(
                        nn.Conv1d(time_cond_channels,
                                  time_cond_channels,
                                  kernel_size=kernel_size,
                                  stride=ratios[i],
                                  padding="same" if ratios[i] == 1 else
                                  (kernel_size) // 2), nn.SiLU()))

        self.up_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()


            
        self.down_layers.append(
            EncoderBlock1D(in_c=in_size + (time_cond_in_channels if cond_type=="input" else 0),
                           out_c=channels[0],
                           time_channels=time_channels,
                           time_cond_channels=time_cond_channels,
                           cond_channels=cond_channels,
                           kernel_size=kernel_size,
                           ratio=ratios[0]))
        comp_ratios = []
        cur_ratio = 1
        for r in ratios:
            cur_ratio *= r
            comp_ratios.append(cur_ratio)

        for i in range(1, n):
            self.down_layers.append(
                EncoderBlock1D(in_c=channels[i - 1],
                               out_c=channels[i],
                               time_channels=time_channels,
                               time_cond_channels=time_cond_channels,
                               cond_channels=cond_channels,
                               kernel_size=kernel_size,
                               ratio=ratios[i],
                               use_self_attn=i >= n - n_attn_layers,
                               cross_attn_channels= context_channels//4 if self.past_cross_attention[i-1] else 0))
            self.up_layers.append(
                DecoderBlock1D(in_c=channels[n - i],
                               out_c=channels[n - i - 1],
                               time_channels=time_channels,
                               time_cond_channels=time_cond_channels,
                               cond_channels=cond_channels,
                               kernel_size=kernel_size,
                               ratio=ratios[n - i],
                               use_self_attn=i <= (n_attn_layers),
                               cross_attn_channels= context_channels//4 if self.past_cross_attention[n-i-1] else 0))

        self.up_layers.append(
            DecoderBlock1D(in_c=channels[0],
                           out_c=out_size,
                           skip_size = in_size + (time_cond_in_channels if cond_type=="input" else 0),
                           time_channels=time_channels,
                           time_cond_channels=time_cond_channels,
                           cond_channels=cond_channels,
                           kernel_size=kernel_size,
                           ratio=ratios[0]))

        self.middle_block = MiddleBlock1D(
            in_c=channels[-1],
            time_channels=time_channels,
            cond_channels=cond_channels,
            time_cond_channels=time_cond_channels,
            kernel_size=kernel_size,
            use_self_attn=n_attn_layers > 0)
        
        
        if self.context_channels>0:
            self.context_emb = Encoder1D(in_size=context_channels,
                 channels=[context_channels, context_channels//2, context_channels//4],
                 ratios=[1, 1, 1],
                 kernel_size=3,
                 cond={},
                 use_tanh = True,
                 average_out=False,
                 upscale_out = False)

    def forward(self, x, time=None, time_cond=None, zsem=None, context=None):
        
        if self.context_channels > 0:
            time_cond, context = time_cond[:,:self.time_cond_in_channels], time_cond[:,self.time_cond_in_channels:]
            context = torch.cat(context.chunk(context.shape[1]//self.context_channels, dim=1), dim=-1)
            
            context = self.context_emb(context)
            
        time = self.time_emb(time).to(x)

        skips = []
        time_conds = []

        if self.time_cond_channels:
            for layer, cond_layer in zip(self.down_layers, self.cond_emb_time):

                if time_cond is not None:
                    time_cond = cond_layer(time_cond)
                

                x, skip = layer(x, time=time, time_cond=time_cond, zsem=zsem, context=context)

                time_conds.append(time_cond)
                skips.append(skip)

            time_cond = self.cond_emb_time[-1](time_cond)

            x = self.middle_block(x, time=time, time_cond=time_cond, zsem=zsem)

            for layer in self.up_layers:
                skip = skips.pop(-1)
                time_cond = time_conds.pop(-1)

                x = layer(x,
                          skip=skip,
                          time=time,
                          time_cond=time_cond,
                          zsem=zsem, context=context)

            return x

        else:
            if time_cond is not None:
                x = torch.cat([x, time_cond], dim=1)
            for layer in self.down_layers:
                x, skip = layer(x, time=time, zsem=zsem, context=context)
                skips.append(skip)

            x = self.middle_block(x, time=time, zsem=zsem)

            for layer in self.up_layers:
                
                skip = skips.pop(-1)

                x = layer(x, skip, time=time, zsem=zsem, context=context)
            return x
