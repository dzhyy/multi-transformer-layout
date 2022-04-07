import math
from torch.optim.lr_scheduler import LambdaLR
    # LambdaLR需要lambda传参，这里写在一起了。
def get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles:float = 0.5,
        last_epoch:int = -1,):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step)/float(max(1,num_warmup_steps)) # 没有到warmup_step时，系数从0逐渐趋于1
        progress = float(current_step - num_warmup_steps)/float(max(1,num_training_steps-num_warmup_steps)) # 趋于1
        return max(0.0, 0.5*(1.0+math.cos(math.pi * float(num_cycles) * 2 * progress))) # 系数从1逐渐趋于0
    return LambdaLR(optimizer, lr_lambda, last_epoch)