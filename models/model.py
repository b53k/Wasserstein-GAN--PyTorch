import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim, d_dim = 16):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        '''nn.ConvTranspose2d(): used in generator ----> 1x1 noise vector with a number of channels to a 
        full sized image [ (n-1)*stride - 2*padding + kernel_size ]
        in convTranspose2d we begin with a 1x1 image with z_dim channels (200). 
        We increase the size of image gradually with a decrease in the number of channels.'''

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels = z_dim, out_channels = d_dim*32 , kernel_size = 4, stride = 1, padding = 0), # 1x1 image 200 channel to 4x4 image 512 channels
            nn.BatchNorm2d(num_features = d_dim*32),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(in_channels = d_dim*32, out_channels = d_dim*16, kernel_size = 4, stride = 2, padding = 1), # to 8x8 image 256 channels
            nn.BatchNorm2d(num_features = d_dim*16),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(in_channels = d_dim*16, out_channels = d_dim*8, kernel_size = 4, stride = 2, padding = 1), # 16x16 image with 128 channels 
            nn.BatchNorm2d(num_features = d_dim*8),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(in_channels = d_dim*8, out_channels = d_dim*4, kernel_size = 4, stride = 2, padding = 1), # 32x32 image with 64 channels 
            nn.BatchNorm2d(num_features = d_dim*4),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(in_channels = d_dim*4, out_channels = d_dim*2, kernel_size = 4, stride = 2, padding = 1), # 64x64 image with 32 channels 
            nn.BatchNorm2d(num_features = d_dim*2),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(in_channels = d_dim*2, out_channels = 3, kernel_size = 4, stride = 2, padding = 1), # 128x128 image with 3 channels 
            nn.Tanh() # squeeze outputs in region -1 to +1 which is suitable for WGAN
        )
    
    def forward(self, noise):
        x = noise.view(-1, self.z_dim, 1, 1)   # output: 128 x 200 x 1 x 1 i.e. batch x channels x 1 x 1
        return self.gen(x)
    
    def gen_noise(self,num, z_dim, device):
        return torch.randn(num, z_dim, device = device) # 128 x 200


# Critic
class Critic(nn.Module):
    def __init__(self, d_dim = 16):
        super(Critic, self).__init__()
        '''nn.conv2d: Used in critic ---> full image to prediction
         new width and height: (n+2*padding - kernel_Size) // stride + 1  where 'n' is previous height and width of image. 
         We're working with image size 128x128'''
        self.critic = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = d_dim, kernel_size = 4, stride = 2, padding = 1),   # 64x64 image & channels = 16
            nn.InstanceNorm2d(d_dim), # It has been found that normalizing by instance works well in Critic model
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels = d_dim, out_channels = d_dim*2, kernel_size = 4, stride = 2, padding = 1),   # 32x32 image & channels = 32
            nn.InstanceNorm2d(d_dim*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels = d_dim*2, out_channels = d_dim*4, kernel_size = 4, stride = 2, padding = 1),   # 16x16 image & channels = 64
            nn.InstanceNorm2d(d_dim*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels = d_dim*4, out_channels = d_dim*8, kernel_size = 4, stride = 2, padding = 1),   # 8x8 image & channels = 128
            nn.InstanceNorm2d(d_dim*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels = d_dim*8, out_channels = d_dim*16, kernel_size = 4, stride = 2, padding = 1),   # 4x4 image & channels = 256
            nn.InstanceNorm2d(d_dim*16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels = d_dim*16, out_channels = 1, kernel_size = 4, stride = 1, padding = 0),   # 1x1 image & channels = 1  (final prediction) 
        )
    
    def forward(self,image):
        # image: batch x channel x height x width
        predict = self.critic(image)  # 128 x 1 x 1 x 1
        return predict.view(len(predict), -1)