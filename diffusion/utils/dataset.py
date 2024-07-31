import torch
import numpy as np
import torchaudio
import gin

from acids_datasets import SimpleDataset


@gin.configurable
def get_dataset(z_length, ae_factor, z_length_t=256, n_past = 0, dataset_type = "choir", db_path = "", time_cond_type = "waveform", use_cache = True, **kwargs):
    
    if dataset_type == "simple_dataset":
        dataset = SimpleDataset(path = db_path, keys=["waveform", "acids_ae"])
        dataset , valset =  torch.utils.data.random_split(dataset, (len(dataset) - 200, 200))
    
        def collate_fn(L):
            x = np.stack([l["waveform"] for l in L])
            x = torch.from_numpy(x).float()
            
            x_ae = np.stack([l["acids_ae"] for l in L])
            x_ae = torch.from_numpy(x_ae).float()
            
            i0 = np.random.randint(0, x_ae.shape[-1] - z_length, x_ae.shape[0])
            x_diff = torch.stack([xc[..., i*ae_factor:i*ae_factor + z_length*ae_factor] for i, xc in zip(i0, x)])
            
            x_diff_ae = torch.stack([xc[..., i:i + z_length] for i, xc in zip(i0, x_ae)])        
            
            
            i1 = np.random.randint(0, x_ae.shape[-1] - z_length, x.shape[0])
            x_toz = torch.stack([xc[..., i:i + z_length] for i, xc in zip(i1, x_ae)])

            return {
                "x": x_diff_ae,
                "x_time_cond": x_diff,
                "x_toz": x_toz,
            }
            
        sampler_train, sampler_val = None, None
   




