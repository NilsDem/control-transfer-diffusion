#EDM Diffusion was adapted from the official implementation https://github.com/NVlabs/edm


from typing import Callable, Optional
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from tqdm import tqdm
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gin
from torch_ema import ExponentialMovingAverage
import os


@gin.configurable
class Base(nn.Module):

    def __init__(self,
                 net=None,
                 encoder=None,
                 encoder_time=None,
                 classifier=None,
                 emb_model=None,
                 data_type="audio",
                 data_prep="waveform_transform", 
                 time_transform = None,
                 drop_values= [-4., -4.],
                 sr = 0,
                 ):
        super().__init__()

        self.net = net
        self.encoder = encoder
        self.encoder_time = encoder_time
        self.classifier = classifier
        self.data_type = data_type
        self.time_transform = time_transform
        self.emb_model = emb_model
        self.data_prep = data_prep
        self.sr = sr
        
        self.time_cond_drop = drop_values[0]
        self.zsem_drop = drop_values[1]
        

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(self,
               x0: torch.Tensor,
               nb_step: int,
               time_cond=None,
               zsem=None,
               guidance: float = 1):
        pass

    def training_step(self, batch, cond=None, time_cond=None):
        pass

    def _backward(self, loss):
        self.accelerator.backward(loss)


    @gin.configurable
    def cfgdrop(self, datas, guidance, values=[-2, -2]):
        draw = torch.rand(datas[0].shape[0])
        test = (draw > guidance)
        test_inv = (~(draw > guidance))
        for i in range(len(datas)):
            if datas[i] is None:
                datas[i] = None
            else:
                datas[i] = self.broadcast_to(test.to(
                    datas[i]), datas[i].shape) * datas[i] + self.broadcast_to(
                        test_inv.to(datas[i]),
                        datas[i].shape) * torch.ones_like(datas[i]) * values[i]

        return datas

    def broadcast_to(self, alpha, shape):
        assert type(shape) == torch.Size
        return alpha.reshape(-1, *((1, ) * (len(shape) - 1)))

    @gin.configurable
    def get_scheduler(self, optimizer, scheduler_type, warmup_steps, decay_steps, decay_max):
        if scheduler_type == "constant":
            return ExponentialLR(optimizer, gamma=1.0)
        elif scheduler_type == "linear":
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return 1.0
                return max(decay_max, (decay_max - 1.0) / (decay_steps) * (current_step-warmup_steps) + 1.0)
            return LambdaLR(optimizer, lr_lambda)
        else:
            raise "Scheduler type not implemented"

    @gin.configurable
    def init_train(self, dataloader, lr):
        params = list(self.net.parameters())
        if self.encoder is not None and self.train_encoder==True:
            params += list(self.encoder.parameters())
        if self.encoder_time is not None and self.train_encoder_time == True:
            params += list(self.encoder_time.parameters())
            
        
        self.opt = AdamW(params, lr=lr, betas=(0.9, 0.999))
        self.scheduler = self.get_scheduler(self.opt)
        self.step = 0

        if self.classifier is not None:
            self.opt_classifier = AdamW(self.classifier.parameters(),
                                        lr=lr,
                                        betas=(0.9, 0.999))
        else:
            self.opt_classifier = None

        self.opt, self.opt_classifier, self.scheduler, dataloader = self.accelerator.prepare(
            self.opt, self.opt_classifier, self.scheduler, dataloader)

        print("init done")
        return dataloader

    @gin.configurable
    @torch.no_grad()
    def prep_data(self, batch, device=None):
        
        x1 = batch["x"].to(device)
        x1_toz = batch["x_toz"].to(device)
        x1_time_cond = batch.get("x_time_cond", None)
        
        if self.time_transform is not None:
            if x1_time_cond is not None:
                time_cond = self.time_transform(x1_time_cond.to(device))
            else:
                time_cond = self.time_transform(x1)
                
        else:
            time_cond = x1_time_cond.to(device)
        
        if x1_toz.shape[1]==1:
            x1 = self.emb_model.encode(x1)
            x1_toz = self.emb_model.encode(x1_toz)
            
        time_cond_add = batch.get("time_cond_add", None)
        time_cond_add = time_cond_add.to(device) if time_cond_add is not None else None
        
        return x1, x1_toz, time_cond, None, time_cond_add
            

    def encode(self, x: torch.Tensor):
        #assert self.encoder is not None
        return self.encoder(x) if self.encoder is not None else None

    def sample_swap(self, x1, time_cond, zsem):

        x0 = torch.randn_like(x1)
        indices = torch.randperm(len(zsem))
        zsem = zsem[indices]

        with torch.no_grad():
            x_swap = self.sample(x0,
                                 time_cond=time_cond,
                                 zsem=zsem,
                                 nb_step=10,
                                 guidance=1,
                                 verbose=False)

        return x_swap, indices


    @gin.configurable
    def fit(self,
            dataset,
            dataloader,
            restart_step,
            model_dir,
            max_steps,
            steps_valid=5000,
            steps_display=100,
            steps_save=25000,
            guidance=0.15,
            validloader=None,
            train_encoder = True,
            train_encoder_time= True,
            use_ema = True,
            device = "cpu",
            use_context = True):
        
        self.train_encoder = train_encoder
        self.train_encoder_time = train_encoder_time
        self.use_ema = use_ema
        
        dataloader = self.init_train(dataloader=dataloader)

        if restart_step > 0:
            state_dict = torch.load(f"{model_dir}/checkpoint" +
                                    str(restart_step) + ".pt",map_location = "cpu")
            
            self.load_state_dict(state_dict["model_state"], strict=False)
            self.opt.load_state_dict(state_dict["opt_state"])
            self.step = restart_step + 1

            print("Restarting from step ", self.step)
        
        if self.use_ema:
            params = list(self.net.parameters()) 
            if self.encoder_time is not None:
                params += list(self.encoder_time.parameters())
            if self.encoder is not None:
                params += list(self.encoder.parameters())
            ema = ExponentialMovingAverage(params, decay=0.999)
            
            

        if self.accelerator.is_main_process:
            logger = SummaryWriter(log_dir=model_dir + "/logs")
            self.tepoch = tqdm(total=max_steps, unit="batch")

        self.accelerator.wait_for_everyone()

        n_epochs = max_steps // len(dataloader) + 1
        losses_sum = {}
        losses_sum_count = {}
        
        
        with open(os.path.join(model_dir, "config.gin"), "w") as config_out:
            config_out.write(gin.operative_config_str())

        for e in range(n_epochs):
            for batch in dataloader:
                x1, x1_toz, time_cond, cond, time_cond_add = self.prep_data(batch, device = device)

                if x1_toz is not None:
                    if self.encoder is not None:
                        if self.train_encoder:
                            zsem = self.encoder(x1_toz)
                        else:
                            with torch.no_grad():
                                zsem = self.encoder(x1_toz)
                    else:
                        zsem = x1_toz
                else:
                    zsem = None

                if self.encoder_time is not None:
                    if self.train_encoder_time:
                        time_cond = self.encoder_time(time_cond)
                    else:
                        with torch.no_grad():
                            time_cond = self.encoder_time(time_cond)
                else:
                    time_cond = time_cond
                    
                if guidance > 0:
                    time_cond_drop = self.cfgdrop([time_cond], guidance, [self.time_cond_drop])[0]
                else:
                    time_cond_drop = time_cond
                    
                
                if cond is not None:
                    zsem = torch.cat((zsem, cond), -1)
                    
                if time_cond_add is not None and use_context==False:
                    time_cond_add =  self.cfgdrop([time_cond_add], 0.2, [self.time_cond_drop])[0]
                    time_cond_full = torch.cat((time_cond_drop, time_cond_add), 1)
                else:
                    time_cond_full = time_cond_drop
                    
                lossdict = self.training_step(x1,
                                              time_cond=time_cond_full,
                                              zsem=zsem,
                                              time_cond_true=time_cond,
                                              zsem_true = zsem)
                                              #context = time_cond_add if use_context == True else None)
                
                if self.use_ema:
                    ema.update()

                lossdict["current_lr"] = self.scheduler.get_last_lr()[0]

                for k in lossdict:
                    losses_sum[k] = losses_sum.get(k, 0.) + lossdict[k]
                    losses_sum_count[k] = losses_sum_count.get(k, 0) + 1

                if self.step % steps_display == 0 and self.accelerator.is_main_process:
                    self.tepoch.set_postfix(loss=losses_sum["Diffusion loss"] /
                                            steps_display)
                    for k in losses_sum:
                        logger.add_scalar('Loss/' + k,
                                          losses_sum[k] /
                                          max(1, losses_sum_count[k]),
                                          global_step=self.step)
                        losses_sum[k] = 0.
                        losses_sum_count[k] = 0
                        
  
                if self.step % steps_valid == 0 and self.step >0:
                    with torch.no_grad():
                        if self.accelerator.is_main_process:
                            self.accelerator.print("Validation")
                            ## Validation
                            lossval = {}
                            for i,batch in enumerate(validloader):
                                x1, x1_toz, time_cond, cond, time_cond_add = self.prep_data(
                                    batch, device=self.device)
                                
        
                                if x1_toz is not None:
                                    zsem = self.encoder(
                                        x1_toz) if self.encoder is not None else x1_toz
                                else:
                                    zsem = None

                                if cond is not None:
                                    zsem = torch.cat((zsem, cond), -1)
                                
                                    
                                time_cond = self.encoder_time(
                                    time_cond
                                ) if self.encoder_time is not None else time_cond
                                
                                #time_cond = time_cond[..., time_cond.shape[-1]//4:-time_cond.shape[-1]//4]
                                
                                if time_cond_add is not None:
                                    time_cond = torch.cat((time_cond, time_cond_add), 1)
                                    

                                lossdict = self.valid_step(x1,
                                                        time_cond=time_cond,
                                                        zsem=zsem)

                                for k in lossdict:
                                    lossval[k] = lossval.get(k, 0.) + lossdict[k]
                                    
                                if i == 100:
                                    break

                            for k in lossval:
                                logger.add_scalar('Loss/valid/' + k,
                                                lossval[k] / 100,
                                                global_step=self.step)
                                
                                
                                
                            with ema.average_parameters():
                                lossval = {}
                                for i,batch in enumerate(validloader):
                                    x1, x1_toz, time_cond, cond, time_cond_add = self.prep_data(
                                        batch, device=self.device)
                                    
            
                                    if x1_toz is not None:
                                        zsem = self.encoder(
                                            x1_toz) if self.encoder is not None else x1_toz
                                    else:
                                        zsem = None

                                    if cond is not None:
                                        zsem = torch.cat((zsem, cond), -1)
                                    
                                        
                                    time_cond = self.encoder_time(
                                        time_cond
                                    ) if self.encoder_time is not None else time_cond
                                    
                                    
                                    if time_cond_add is not None:
                                        time_cond = torch.cat((time_cond, time_cond_add), 1)
                                        

                                    lossdict = self.valid_step(x1,
                                                            time_cond=time_cond,
                                                            zsem=zsem)

                                    for k in lossdict:
                                        lossval[k] = lossval.get(k, 0.) + lossdict[k]
                                        
                                    if i == 100:
                                        break

                                for k in lossval:
                                    logger.add_scalar('Loss/valid_EMA/' + k,
                                                    lossval[k] / 100,
                                                    global_step=self.step)
                            

                            ## Data out

                            x1 = x1[:6].to(self.device)
                            time_cond = time_cond[:6] if time_cond is not None else None
                            zsem = zsem[:6] if zsem is not None else None
                            x0 = self.sample_prior(x1.shape)
                            

                            x1_rec = self.sample(x0,
                                                nb_step=40,
                                                time_cond=time_cond,
                                                zsem=zsem)

                            
                            audio_true = self.emb_model.decode(x1).cpu()
                            audio_rec = self.emb_model.decode(x1_rec).cpu()

                            audio = torch.cat(
                                (audio_true,
                                torch.zeros(audio_true.shape[0], 1,
                                            20000), audio_rec), -1)
                            
                            audio = audio.reshape(-1)

                            logger.add_audio("audio",
                                            audio.detach(),
                                            global_step=self.step,
                                            sample_rate=self.sr)
                    
                if self.step % steps_save == 0:
                    self.accelerator.wait_for_everyone()
                    self.save_model(model_dir, ema = ema if self.use_ema else None)

                if self.accelerator.is_main_process:
                    self.tepoch.update(1)

                self.step += 1

    def save_model(self, model_dir, ema=None):
        model = self.accelerator.unwrap_model(self)
        d = {
            "model_state": {k:v for k,v in model.state_dict().items() if "emb_model" not in k},
            "opt_state": self.opt.state_dict()
        }

        self.accelerator.save(
            d, model_dir + "/checkpoint" + str(self.step) + ".pt")
        
        if ema is not None:
            with ema.average_parameters():
                d = {
                    "model_state": {k:v for k,v in model.state_dict().items() if "emb_model" not in k},
                }

                self.accelerator.save(
                    d, model_dir + "/checkpoint" + str(self.step) + "_EMA.pt")

    def load_model(self, path):
        pass


class EDM(Base):

    def __init__(
        self,
        *,
        sigma_min: float = 0.002,
        sigma_max: float = 80.,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        rho: float = 7,
        time_cond_warmup = 0,
        **basekwargs,
    ):
        super().__init__(**basekwargs)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p_mean = p_mean
        self.p_std = p_std
        self.rho = rho
        self._global = 0
        self._epoch = 0
        self.time_cond_warmup = time_cond_warmup
        return

    def estimate_std(
        self,
        loader = None,
        n_batches: int = 128,
    ) -> float:
        xlist = []
        print("Estimating sigma...")
        
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            x = batch["x"]
            xlist.append(x)
        x = torch.cat(xlist, dim=0)
        sigma = x.std()
        print(f"setting sigma_data to {sigma}")
        return sigma.item()

    def _get_scalings(self, sigma: torch.Tensor):
        c_skip = self.sdata**2 / (sigma**2 + self.sdata**2)
        c_out = sigma * self.sdata / (sigma**2 + self.sdata**2).sqrt()
        c_in = 1 / (self.sdata**2 + sigma**2).sqrt()
        c_noise = 0.25 * sigma.log()
        
        return c_skip, c_out, c_in, c_noise

    def _get_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.sdata**2) / (sigma * self.sdata)**2

    def _get_sigma(self, data_shape: int) -> torch.Tensor:
        z = torch.randn(data_shape, device=self.device)
        sigma = (z * self.p_std + self.p_mean).exp()
        return sigma

    def _model_forward(self,
                       *,
                       x: torch.Tensor,
                       noisy: torch.Tensor,
                       sigma: torch.Tensor,
                       time_cond: Optional[torch.Tensor],
                       zsem: Optional[torch.Tensor],
                       guidance: float = 1) -> torch.Tensor:
        return self._fwd(x=noisy,
                         sigma=sigma,
                         time_cond=time_cond,
                         zsem=zsem,
                         guidance=guidance)

    def _fwd(self,
             x: torch.Tensor,
             sigma: torch.Tensor,
             time_cond: Optional[torch.Tensor] = None,
             zsem: Optional[torch.Tensor] = None,
             guidance: float = 1,
             guidance_type: str = "both"):

        c_skip, c_out, c_in, c_noise = self._get_scalings(sigma)
        f_xy = self.net(c_in * x,
                        time=c_noise.reshape(-1),#.squeeze(),
                        time_cond=time_cond,
                        zsem=zsem)

        if guidance != 1:
           # print("using guidance")
            if guidance_type == "time_cond" or guidance_type == "both":
                time_cond = self.time_cond_drop * torch.ones_like(
                    time_cond) if time_cond is not None else None

            if guidance_type == "zsem" or guidance_type == "both":
                zsem = self.zsem_drop * torch.ones_like(zsem) if zsem is not None else None

            f_x = self.net(c_in * x,
                           time=c_noise.reshape(-1),
                           time_cond=time_cond,
                           zsem=zsem)
            f_xy = (f_x * (1 - guidance) + guidance * f_xy)

        d = c_skip * x + c_out * f_xy
        return d

    def sample_prior(self, x0_shape):
        return torch.randn(x0_shape).to(self.device)



    def training_step(self, x1: torch.Tensor,
                      time_cond: Optional[torch.Tensor],
                      zsem: Optional[torch.Tensor], **kwargs) -> torch.Tensor:


        if self.step < self.time_cond_warmup:
            time_cond = torch.zeros_like(time_cond)
            
        b, *_ = x1.shape
        data_shape = (b, *([1] * (len(x1.shape) - 1)))
        sigma = self._get_sigma(data_shape)
        x_noisy = x1 + torch.randn_like(x1) * sigma
        d = self._model_forward(x=x1,
                                noisy=x_noisy,
                                sigma=sigma,
                                time_cond=time_cond,
                                zsem=zsem)
        weight = self._get_weight(sigma)
        loss = weight * ((d - x1)**2)

        loss = loss.mean()

        self.opt.zero_grad()
        self._backward(loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0, norm_type=2.0)
        #self.clip_fn(self.net.parameters())
        self.opt.step()
        self.scheduler.step()

        return {"Diffusion loss": loss.item()}

    @torch.no_grad()
    def valid_step(self, x1: torch.Tensor, time_cond: Optional[torch.Tensor],
                   zsem: Optional[torch.Tensor]):
        b, *_ = x1.shape
        data_shape = (b, *([1] * (len(x1.shape) - 1)))
        sigma = self._get_sigma(data_shape)
        x_noisy = x1 + torch.randn_like(x1) * sigma
        
        d = self._model_forward(x=x1,
                                noisy=x_noisy,
                                sigma=sigma,
                                time_cond=time_cond,
                                zsem=zsem)
        weight = self._get_weight(sigma)
        loss = weight * ((d - x1)**2)

        loss = loss.mean()
        return {"Diffusion loss": loss.item()}

    @torch.no_grad()
    def sample(self,
               x0: Optional[torch.Tensor] = None,
               nb_step=  256,
               time_cond: Optional[torch.Tensor] = None,
               zsem: Optional[torch.Tensor] = None,
               sigma_max=None,
               sigma_min=None,
               guidance=1,
               guidance_type="both",
               verbose=True,
               special_fwd = None):
        sigma_max = sigma_max if sigma_max is not None else self.sigma_max
        sigma_min = sigma_min if sigma_min is not None else self.sigma_min

        return self._heun_sample(
            base=x0,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            rho=self.rho,
            fwd_fn=special_fwd if special_fwd is not None else self._fwd,
            nb_step=nb_step,
            time_cond = time_cond,
            zsem = zsem,
            guidance =guidance,
            guidance_type =guidance_type,
            verbose=verbose,
        )

    def _heun_sample(
        self,
        base: torch.Tensor,
        time_cond ,
        zsem,
        guidance,
        guidance_type,
        sigma_max: float,
        sigma_min: float,
        rho: float,
        fwd_fn,
        nb_step,
        verbose: bool = False,
    ):
        
        S_churn = 0.
        S_min = 0.0
        S_max= float("inf")
        S_noise = 1.0
        
        """https://github.com/NVlabs/edm/blob/main/generate.py#L25"""
        step_indices = torch.arange(nb_step,
                                    dtype=torch.float32,
                                    device=base.device)
        t_steps = (sigma_max**(1 / rho) + step_indices / (nb_step - 1) *
                   (sigma_min**(1 / rho) - sigma_max**(1 / rho)))**rho
        t_steps = torch.cat(
            [torch.as_tensor(t_steps),
             torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        x_next = base * t_steps[0]
        batch_size = x_next.shape[0]
        
        

        if verbose == True:
            pbar = tqdm(
                list(enumerate(zip(t_steps[:-1], t_steps[1:]))),
                unit="step",
            )
        else:
           
            pbar = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
        for i, (t_cur, t_next) in pbar:  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / nb_step,
                        torch.sqrt(torch.tensor(2.)) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (
                t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            #ndim = len(x_hat.shape)-1
            _t_hat = t_hat.repeat(batch_size).view(
                (batch_size, *(1 for _ in range(x_hat.ndim - 1))))
            
      
            denoised = fwd_fn(x=x_hat, sigma=_t_hat,zsem = zsem, time_cond=time_cond, guidance=guidance, guidance_type=guidance_type)
           
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < nb_step - 1:
                _t_next = t_next.repeat(batch_size).view(
                    (batch_size, *(1 for _ in range(x_next.ndim - 1))))
                denoised = fwd_fn(x=x_next, sigma=_t_next,zsem = zsem, time_cond=time_cond, guidance=guidance, guidance_type=guidance_type)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur +
                                                     0.5 * d_prime)

        return x_next


@gin.configurable
class EDM_ADV(EDM):

    def __init__(
        self,
        *,
        reg_classifier: float,
        warmup_classifier: int,
        warmup_timbre : int,
        #reg_timbre: float,
        sigma_min: float = 0.002,
        sigma_max: float = 80.,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        rho: float = 7,
        sdata: float,
        **basekwargs,
    ):
        super().__init__(**basekwargs)
        self.reg_classifier = reg_classifier
        self.warmup_classifier = warmup_classifier
        self.warmup_timbre = warmup_timbre
        # self.reg_timbre = reg_timbre
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p_mean = p_mean
        self.p_std = p_std
        self.rho = rho
        self.sdata = sdata
        return

    def triplet_loss(self, anchor, pos, neg, margin=0.4):
        if len(anchor.shape) == 2:
            dist_ap = (torch.linalg.norm(anchor - pos, dim=1)).unsqueeze(-1)
            dist_an = (torch.linalg.norm(anchor - neg, dim=1)).unsqueeze(-1)
        elif len(anchor.shape) == 3:
            dist_ap = (torch.linalg.norm(anchor - pos,
                                         dim=1)).mean(-1).unsqueeze(-1)
            dist_an = (torch.linalg.norm(anchor - neg,
                                         dim=1)).mean(-1).unsqueeze(-1)

        dist_ap, dist_an = dist_ap / np.sqrt(
            anchor.shape[1]), dist_an / np.sqrt(anchor.shape[1])
        return torch.nn.functional.relu(dist_ap - dist_an + margin)

    def training_step(self, x1: torch.Tensor,
                      time_cond: Optional[torch.Tensor],
                      time_cond_true: Optional[torch.Tensor],
                      zsem: Optional[torch.Tensor],
                      zsem_true: Optional[torch.Tensor]) -> torch.Tensor:

        if self.warmup_classifier > 0:
            reg_classifier = min(
                self.reg_classifier *
                ((self.step - self.warmup_classifier) / self.warmup_classifier),
                self.reg_classifier) if self.step > self.warmup_classifier else 0.
        else:
            reg_classifier = self.reg_classifier

        if self.step > self.warmup_classifier and self.step % 3 == 0 and self.classifier is not None:
            
            
            classifier_loss = torch.nn.functional.mse_loss(self.classifier(
                time_cond_true.detach()),
                                                           zsem_true.detach(),
                                                           reduction='mean')

            self.opt_classifier.zero_grad()
            self._backward(classifier_loss)
            self.opt_classifier.step()

            return {
                "Classifier loss": classifier_loss.item(),
            }
        else:
            b, *_ = x1.shape
            data_shape = (b, *([1] * (len(x1.shape) - 1)))
            sigma = self._get_sigma(data_shape)
            x_noisy = x1 + torch.randn_like(x1) * sigma

            if self.step < self.warmup_timbre:
                time_cond = self.time_cond_drop * torch.ones_like(time_cond)


            if self.step<self.warmup_classifier or self.classifier is None:
                classifier_loss = torch.tensor(0.)

            else:
                classifier_loss = torch.nn.functional.mse_loss(
                    self.classifier(time_cond_true),
                    zsem_true.detach(),
                    reduction='mean')

            d = self._model_forward(x=x1,
                                    noisy=x_noisy,
                                    sigma=sigma,
                                    time_cond=time_cond,
                                    zsem=zsem)
            weight = self._get_weight(sigma)
            diffloss = weight * ((d - x1)**2)
            diffloss = diffloss.mean()


            loss = diffloss - reg_classifier * classifier_loss

            self.opt.zero_grad()
            self._backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(),
                                           1.0,
                                           norm_type=2.0)
            self.opt.step()
            self.scheduler.step()

            return {
                "Diffusion loss": diffloss.item(),
                "Classifier loss": classifier_loss.item(),
                "Regularisation weight": reg_classifier
            }
