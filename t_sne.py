import copy
import json
import os
import warnings

import torch
from absl import app, flags
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import random

from diffusion import GaussianDiffusion_distillation_Trainer, GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score


FLAGS = flags.FLAGS

flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
flags.DEFINE_bool('distill', False, help='perform knowledge distillation')

# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 1e-5, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def t_sne():
    # Load pretrained UNet model
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    model.load_state_dict(ckpt['ema_model'])
    model.eval()

    # Load CIFAR-10 dataset and randomly select 128 samples
    dataset = CIFAR10(
        root='./data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    # Select 128 random samples
    indices = random.sample(range(len(dataset)), 128)
    samples = torch.stack([dataset[i][0] for i in indices]).to(device)

    # Get model output for the selected samples
    with torch.no_grad():
        outputs = model(samples)

    # Flatten the outputs for t-SNE (usually we take features before the final layer)
    # Assuming model outputs are in shape [batch_size, channels, height, width]
    outputs_flat = outputs.view(outputs.size(0), -1).cpu().numpy()

    # Apply t-SNE to the outputs
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(outputs_flat)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=10, cmap='tab10')
    plt.title('t-SNE of UNet Outputs on CIFAR-10 Samples')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()





def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.t_sne:
        t_sne()
    if not FLAGS.train and not FLAGS.eval and not FLAGS.distill:
        print('Add --train, --eval and/or --distill to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
