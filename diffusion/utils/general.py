
import torch
class DummyAccelerator():
    def __init__(self, device):
        self.device = device 
        self.num_processes = 1
        self.is_main_process = True
        pass
    
    def backward(self, loss):
        loss.backward()
        
    def prepare(self,*args):
        return args
    
    def wait_for_everyone(self):
        pass
    
    def print(self, *arg):
        print(arg)
    
    def unwrap_model(self,model):
        return model
    
    def save(self, d, path):
        torch.save(d, path)
    
    