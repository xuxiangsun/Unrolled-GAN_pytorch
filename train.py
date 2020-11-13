# This project mainly reference https://github.com/andrewliao11/unrolled-gans
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from tqdm import tqdm_notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class data_generator(object):
    def __init__(self):

        n = 8
        radius = 2
        std = 0.02
        delta_theta = 2*np.pi / n

        centers_x = []
        centers_y = []
        for i in range(n):
            centers_x.append(radius*np.cos(i*delta_theta))
            centers_y.append(radius*np.sin(i*delta_theta))

        centers_x = np.expand_dims(np.array(centers_x), 1)
        centers_y = np.expand_dims(np.array(centers_y), 1)

        p = [1./n for _ in range(n)]

        self.p = p
        self.size = 2
        self.n = n
        self.std = std
        self.centers = np.concatenate([centers_x, centers_y], 1)

    # switch to random distribution (harder)
    def random_distribution(self, p=None):
        if p is None:
            p = [np.random.uniform() for _ in range(self.n)]
            p = p / np.sum(p)
        self.p = p

    # switch to uniform distribution
    def uniform_distribution(self):
        p = [1./self.n for _ in range(self.n)]
        self.p = p

    def sample(self, N):
        n = self.n
        std = self.std
        centers = self.centers

        ith_center = np.random.choice(n, N,p=self.p)
        sample_centers = centers[ith_center, :]
        sample_points = np.random.normal(loc=sample_centers, scale=std)
        return sample_points.astype('float32')

plt.style.use('ggplot')

# define a plot function to visualize the distribution of the generated data and the real data
def plot(points, title):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=10, c='b', alpha=0.5)
    plt.scatter(dset.centers[:, 0], dset.centers[:, 1], s=100, c='g', alpha=0.5)
    plt.title(title)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.savefig('./training_results/' + title + '.png')
    # plt.show()
    plt.close() # Don't forget this command! Or the new generated picture will be the addition of all the old pictures.

# define a noise sampler
def noise_sampler(N, z_dim):
    return np.random.normal(size=[N, z_dim]).astype('float32')


# define a generator and a discriminator, both of which are all composed of full-connected layers
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = F.tanh
        #self.activation_fn = F.relu

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return self.map3(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = F.relu

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return F.sigmoid(self.map3(x))

# define the training loop of the discriminator
def d_loop(d_gen_input=None):
    # 1. Train D on real+fake
    d_optimizer.zero_grad()

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(minibatch_size))
    d_real_data = d_real_data.cuda()
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    target = target.cuda()
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    if d_gen_input is None:
        d_gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    d_gen_input = d_gen_input.cuda()
    
    with torch.no_grad():
        d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    target = target.cuda()
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake
    
    d_loss = d_real_error + d_fake_error
    d_loss.backward(create_graph=True)
    d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
    return d_real_error.cpu().item(), d_fake_error.cpu().item()

# define the training loop of the generator
def g_loop():
    g_optimizer.zero_grad()

    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    gen_input = gen_input.cuda()

    # unrolling training phase
    if unrolled_steps > 0:
        temp_state_dict = D.state_dict()
        if hasattr(temp_state_dict, '_metadata'):
            del temp_state_dict._metadata
        for i in range(unrolled_steps):
            d_loop(d_gen_input=gen_input)
    
    g_fake_data = G(gen_input)
    dg_fake_decision = D(g_fake_data)
    target = torch.ones_like(dg_fake_decision)
    target = target.cuda()
    g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
    g_error.backward()
    g_optimizer.step()  # Only optimizes G's parameters
    
    if unrolled_steps > 0:
        D.load_state_dict(temp_state_dict)
        del temp_state_dict
    return g_error.cpu().item()


# define the generation function of the generator in testing phase.
def g_sample():
    with torch.no_grad():
        gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
        gen_input = gen_input.cuda()
        g_fake_data = G(gen_input)
        return g_fake_data.cpu().numpy()


# define plot function in testing phase
def plot_samples(samples):
    xmax = 5
    cols = len(samples)
    bg_color  = seaborn.color_palette('Greens', n_colors=256)[0]
    plt.figure(figsize=(2*cols, 2))
    for i, samps in enumerate(samples):
        if i == 0:
            ax = plt.subplot(1, cols, 1)
        else:
            plt.subplot(1, cols, i+1, sharex=ax, sharey=ax)
        ax2 = seaborn.kdeplot(samps[:, 0], samps[:, 1], shaded=True, cmap='Greens', n_levels=20, clip=[[-xmax,xmax]]*2)
        plt.xticks([])
        plt.yticks([])
        plt.title('step %d'%(i*log_interval))
    
    ax.set_ylabel('%d unrolling steps'% unrolled_steps)
    plt.gcf().tight_layout()
    plt.savefig(prefix + '.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    dset = data_generator()
    dset.random_distribution()
    # customize hyperparameters
    z_dim = 256 # dimention of latent code
    g_inp = z_dim # the input union of the first layer of the generator
    g_hid = 128 # the hidden union of the full-connected layers in the generator
    g_out = dset.size # the output union of the last layer of the generator

    d_inp = g_out # the input union of the last layer of the generator
    d_hid = 128 # the hidden union of the full-connected layers in the discriminator
    d_out = 1 # the output union of the last layer of the discriminator

    minibatch_size = 512 # batch size

    unrolled_steps = 10 # unrolling steps
    d_learning_rate = 1e-4 # learning rate of discriminator
    g_learning_rate = 1e-3# learning rate of generator
    optim_betas = (0.5, 0.999) # parameters of adam optimizers
    num_iterations = 6000 # training iterations
    log_interval = 300 # 
    d_steps = 1  
    g_steps = 1
    prefix = "unrolled_steps-{}-prior_std-{:.2f}".format(unrolled_steps, np.std(dset.p))


    G = Generator(input_size=g_inp, hidden_size=g_hid, output_size=g_out).cuda()
    D = Discriminator(input_size=d_inp, hidden_size=d_hid, output_size=d_out).cuda()
    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)
    samples = []
    for it in tqdm_notebook(range(num_iterations)):
        d_infos = []
        for d_index in range(d_steps):
            d_info = d_loop()
            d_infos.append(d_info)
        d_infos = np.mean(d_infos, 0)
        d_real_loss, d_fake_loss = d_infos
        
        g_infos = []
        for g_index in range(g_steps):
            g_info = g_loop()
            g_infos.append(g_info)
        g_infos = np.mean(g_infos)
        g_loss = g_infos
        
        if it % log_interval == 0:
            fake_data = g_sample()
            samples.append(fake_data)
            if not os.path.exists('./training_results'):
                os.makedirs('./training_results')
            plot(fake_data, title='[{}] Iteration {}'.format(prefix, it))
            print(d_real_loss, d_fake_loss, g_loss)
            # del fake_data
# 
    # plot_samples(samples)