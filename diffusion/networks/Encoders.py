import torch
import torch.nn as nn
import cached_conv as cc
import gin


def __prepare_scriptable__(self):
    for hook in self._forward_pre_hooks.values():
        # The hook we want to remove is an instance of WeightNorm class, so
        # normally we would do `if isinstance(...)` but this class is not accessible
        # because of shadowing, so we check the module name directly.
        # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
        if hook.__module__ == "torch.nn.utils.weight_norm" and hook.__class__.__name__ == "WeightNorm":
            print("Removing weight_norm from %s", self.__class__.__name__)
            torch.nn.utils.remove_weight_norm(self)
    return self

def normalization(module: nn.Module, mode: str = 'weight_norm'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        layer = torch.nn.utils.weight_norm(module)
        layer.__prepare_scriptable__ = __prepare_scriptable__.__get__(layer)
        return layer
    else:
        raise Exception(f'Normalization mode {mode} not supported')
    
@gin.configurable
class V2ConvBlock1D(nn.Module):

    def __init__(self, in_c, out_c, kernel_size, act=nn.SiLU, res=True, cumulative_delay=0):
        super().__init__()
        self.res = res

        conv1 = normalization(cc.Conv1d(in_c,
                          out_c,
                          kernel_size=kernel_size,
                          padding=cc.get_padding(kernel_size)))
        
        conv2 = normalization(cc.Conv1d(out_c,
                          out_c,
                          kernel_size=kernel_size,
                          padding=cc.get_padding(kernel_size), cumulative_delay=conv1.cumulative_delay))

        self.gn1 = nn.BatchNorm1d(in_c)
        self.gn2 = nn.BatchNorm1d(out_c)
        act = act
        self.dp = nn.Dropout(p=0.1)
    
        #net = [self.gn1, act(), conv1, self.gn2, act(), self.dp, conv2]
        net = [self.gn1, act(), conv1, self.gn2, act(), self.dp, conv2]
        net = cc.CachedSequential(*net)
        
        additional_delay = net.cumulative_delay 
        
        self.net = cc.AlignBranches(
            net,
            nn.Identity(),
            delays=[additional_delay, 0],
        )
        self.cumulative_delay = additional_delay + cumulative_delay
        
        

    def forward(self, x):
        x, res = self.net(x)
        return x + res

@gin.configurable
class V2EncoderBlock1D(nn.Module):

    def __init__(self, in_c, tot_cond_channels, out_c, kernel_size, ratio, cumulative_delay=0):
        super().__init__()
        conv = V2ConvBlock1D(in_c + tot_cond_channels, in_c, kernel_size, cumulative_delay=cumulative_delay)
        
        if ratio!=1:
            pool = normalization(cc.Conv1d(in_c,
                                out_c,
                                kernel_size=2 * ratio,
                                stride=ratio,
                                padding=cc.get_padding(2 * ratio, ratio), cumulative_delay=conv.cumulative_delay))
        else:
            pool = normalization(cc.Conv1d(in_c,
                                out_c,
                                kernel_size=1,
                                stride=ratio,
                                padding=(0,0), cumulative_delay=conv.cumulative_delay))
        
        
        net = [conv, pool]
        
        
        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net[-1].cumulative_delay
        
    def forward(self, x):
        return self.net(x)

@gin.configurable
class Encoder1D(nn.Module):

    def __init__(self,
                 in_size=1,
                 channels=[64, 128, 128, 256, 256],
                 ratios=[2,2,2,2,2],
                 kernel_size=5,
                 cond={},
                 use_tanh = True,
                 average_out=False,
                 upscale_out = False):
        super().__init__()
        
        self.use_tanh = use_tanh
        self.channels = channels
        self.average_out = average_out
        ratios = [1] + ratios
        n = len(self.channels)

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


        net = []

        net.append(
            V2EncoderBlock1D(in_size,
                             c_channels,
                             channels[0],
                             kernel_size,
                             ratio=ratios[0]))

        for i in range(1, n):
            net.append(
                V2EncoderBlock1D(channels[i - 1], c_channels, channels[i],
                                 kernel_size, ratios[i], cumulative_delay = net[-1].cumulative_delay))
            
            
        net.append(V2ConvBlock1D(channels[-1] + c_channels,
                                          channels[-1], kernel_size, cumulative_delay=net[-1].cumulative_delay))
        
        self.net = cc.CachedSequential(*net)
        
        total_ratio = 1
        for r in ratios:
            total_ratio*=r
        self.total_ratio = float(total_ratio)
        
        self.upscale_out = upscale_out

    @torch.jit.export
    def forward(self, x):

        x = self.net(x)

        if self.average_out:
            x = torch.mean(x, dim=-1)

        if self.use_tanh:
            x =  torch.tanh(x)
            
        if self.upscale_out: 
            x = torch.nn.functional.upsample(x, scale_factor = self.total_ratio)
        
        return x
    
    
@gin.configurable
class LinearEncoder(nn.Module):

    def __init__(self,
                 in_size=512,
                 channels=[512, 1024, 1024, 256, 8],
                 drop_out = 0.15):
                 #out_fn=nn.Identity(),
                 #**kwargs):
        super().__init__()
        module_list = []
        module_list.append(nn.Linear(in_size, channels[0]))

        for i in range(len(channels)-1):
            module_list.append(nn.SiLU())
            module_list.append(nn.Dropout(p=drop_out)) 
            module_list.append(nn.Linear(channels[i], channels[i+1]))
        
        self.net = nn.Sequential(*module_list)
        
        #self.out_fn = out_fn
        
    def forward(self,x):
        return torch.tanh(self.net(x))
        
    