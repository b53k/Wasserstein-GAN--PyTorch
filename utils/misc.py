import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

show_step = 35  # Show status every 35 step
save_step = 35  # save checkpoint every 35 step

# Gradient Penalty
def get_gp(real, fake, crit, alpha, gamma = 10):
    x = alpha*real + (1-alpha)*fake # linear interpolation between a real and a fake image: batch_size x channels x height x width
    y = crit(x) # 128 x 1
    # gradient = dy/dx computes gradient of output w.r.t input
    gradient = torch.autograd.grad(
        inputs = x,
        outputs = y,
        grad_outputs = torch.ones_like(y),
        retain_graph = True,
        create_graph = True
    )[0]  # 128 x 3 x 128 x 128

    gradient = gradient.view(len(gradient),-1)
    gradient_norm = gradient.norm(p = 2, dim =1)  # p-norm = 2 along horizontal dimension of 1
    gradient_penalty = gamma*((gradient_norm-1)**2).mean()  # we calcuate the mean because we're trying to get the expected value
    
    return gradient_penalty

# show real or fake images in grid
def show(tensor, num = 25):
    data = tensor.detach().to('cpu')
    grid = make_grid(data[:num], nrow = 5).permute(1,2,0)
    #plt.imshow(grid.clip(0,1))
    #plt.show()
    return grid