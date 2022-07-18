'''Training WGAN'''
import argparse
import torch, os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.model import Generator, Critic
from utils.dataset import Dataset
from utils.misc import get_gp, show, show_step, save_step

# parse argument
parser = argparse.ArgumentParser(description = 'WGAN Trainer...Need Hyperparemeters')
parser.add_argument('--batch-size', type = int, default = 128, metavar = 'N', help = 'input batch size for training (default: 128)')
parser.add_argument('--n-epochs', type = int, default = 10000, metavar = 'N', help = 'input number of epochs for training (default: 10000)')
parser.add_argument('--z-dim', type = int, default = 200, metavar = 'N', help = 'input latent dimension of Generator (default: 200)')
parser.add_argument('--lr', type = float, default = 1e-4, metavar = 'N', help = 'input learning rate (default: 1e-4)')
parser.add_argument('--noreload', action = 'store_true', help = 'previously saved model will not be reloaded')
parser.add_argument('--logdir', type = str, help = 'Directory where results are logged')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(2022)

# For numeric stabiity
torch.backends.cudnn.benchmark = True

# Hyper-parameters
args = parser.parse_args()
batch_size = args.batch_size
n_epochs = args.n_epochs
z_dim = args.z_dim
lr = args.lr

# Dataset
path = './data/img_align_celeba/'
ds = Dataset(path, im_size = 128, lim = 10)

# DataLoader
dataloader = DataLoader(dataset = ds, batch_size = batch_size, shuffle = True, drop_last = False)

# Models
gen = Generator(z_dim).to(device)
crit = Critic().to(device)

# Optimizers
gen_opt = torch.optim.Adam(gen.parameters(), lr = lr, betas = [0.6, 0.9])
crit_opt = torch.optim.Adam(crit.parameters(), lr = lr, betas = [0.6, 0.9])

# Save and Load Checkpoint
root_path = './checkpoint/'

def save_checkpoint(name):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    torch.save({
        'Epoch':epoch,
        'Model_State_Dict':gen.state_dict(),
        'Optimizer_State_Dict': gen_opt.state_dict(),
        'Loss': loss_gen
    }, f'{root_path}Generator-{name}.pkl')

    torch.save({
        'Epoch': epoch,
        'Model_State_Dict': crit.state_dict(),
        'Optimizer_State_Dict': crit_opt.state_dict(),
        'Loss': loss_critic
        }, f'{root_path}Critic-{name}.pkl')
    
    print ('Checkpoint Saved')

def load_checkpoint(name):
    checkpoint = torch.load(f'{root_path}Generator-{name}.pkl')
    gen.load_state_dict(checkpoint['Model_State_Dict'])
    gen_opt.load_state_dict(checkpoint['Optimizer_State_Dict'])
    epoch = checkpoint['Epoch']
    loss_gen = checkpoint['Loss']

    checkpoint = torch.load(f'{root_path}Critic-{name}.pkl')
    crit.load_state_dict(checkpoint['Model_State_Dict'])
    crit_opt.load_state_dict(checkpoint['Optimizer_State_Dict'])
    epoch = checkpoint['Epoch']
    loss_crit = checkpoint['Loss']
    
    print ('')
    print ('Checkpoint Loaded with the following parameters:')
    print ('Epoch:{} Generator Loss: {:.4f} Critic Loss: {:.4f}'.format(epoch,loss_gen,loss_crit))
    print ('==============================================================================')
    print ('')


# Training Loop
cur_step = 0
crit_cycles = 5
gen_losses, crit_losses = [], []

reload_file_gen = os.path.join(root_path,'Generator-latest.pkl')
reload_file_crt = os.path.join(root_path,'Critic-latest.pkl')

if not args.noreload and os.path.exists(reload_file_gen and reload_file_crt):
    load_checkpoint('latest')

sample_img_dir = os.path.join(args.logdir, 'sample_imgs/')
if not os.path.exists(sample_img_dir):
    os.makedirs(sample_img_dir)

training_log_dir = os.path.join(args.logdir, 'loss_charts/')
if not os.path.exists(training_log_dir):
    os.makedirs(training_log_dir)

# since we're training these guys
gen.train()
crit.train()

pbar = tqdm(total = len(dataloader)*n_epochs, position = 0, leave = True)
for epoch in range(n_epochs):
    #pbar = tqdm(total = len(dataloader), desc = 'Epoch: {}'.format(epoch), position = 0, leave = True)
    for real, _ in dataloader:
        cur_bs = len(real)
        real = real.to(device)

        # Critic
        loss_critic = 0
        '''President Biden says that training Critic for more number of times than Generator increases the performance of WGAN'''
        for _ in range(crit_cycles):
            crit_opt.zero_grad()
            noise = gen.gen_noise(cur_bs, z_dim, device = device)
            fake = gen.forward(noise)
            crit_fake_pred = crit.forward(fake.detach())
            crit_real_pred = crit.forward(real)

            # We need to get the gradient penalty before computing the loss
            alpha = torch.rand(cur_bs,1,1,1, device = device, requires_grad = True)  # 128 random numbers form a uniform distribution betwwen 0 and 1
            gp = get_gp(real, fake.detach(), crit, alpha)

            cr_loss = crit_fake_pred.mean() - crit_real_pred.mean() + gp

            loss_critic += cr_loss.item() / crit_cycles

            cr_loss.backward(retain_graph = True)
            crit_opt.step()
            
        crit_losses += [loss_critic]

        # Generator
        gen_opt.zero_grad()
        noise = gen.gen_noise(cur_bs,z_dim, device = device)
        fake = gen.forward(noise)
        crit_fake_pred = crit.forward(fake)

        loss_gen = -crit_fake_pred.mean()
        loss_gen.backward(retain_graph = True)
        gen_opt.step()

        gen_losses += [loss_gen.item()]

        # Save Checkpoint
        if (cur_step % save_step == 0 and cur_step > 0):
            print ('')
            print ('==============================================================================')
            print ('Saving Checkpoint with the following parameters:')
            print ('Epoch: {} Generator Loss: {:.4f} Critic Loss: {:.4f}'.format(epoch, loss_gen, loss_critic))
            print ('==============================================================================')
            print ('')
            save_checkpoint('latest') # Overrides

        # Status
        if (cur_step % show_step == 0 and cur_step > 0):
            fig = plt.figure()
            fig.add_subplot(1,2,1)
            plt.title('Fake Image')
            plt.imshow(show(fake))
            fig.add_subplot(1,2,2)
            plt.title('Real Image')
            plt.imshow(show(real))
            plt.savefig(f'{sample_img_dir}Sample-{cur_step}.jpg')
            plt.close(fig)

            gen_mean = sum(gen_losses[-show_step:]) / show_step
            crit_mean = sum(crit_losses[-show_step:]) / show_step
            print ('')
            print ('==============================================================================')
            print('Epoch: {} Generator Loss: {:.4f} Critic Loss: {:.4f}'.format(epoch, gen_mean, crit_mean))
            print ('==============================================================================')
            print ('')

            plt.figure()
            plt.plot(range(len(gen_losses)), torch.Tensor(gen_losses), label = 'Generator Loss')
            plt.plot(range(len(crit_losses)), torch.Tensor(crit_losses), label = 'Critic Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.ylim(-80,80)
            plt.savefig(f'{training_log_dir}Loss_Graph-{cur_step}.jpg')
            plt.close()
        cur_step+=1
    
    pbar.set_postfix_str({'Gen Loss':'{0:.4f}'.format(sum(gen_losses)/len(gen_losses)), 'Crit Loss':'{0:.4f}'.format(sum(crit_losses)/len(crit_losses))})
    pbar.update()
pbar.close()