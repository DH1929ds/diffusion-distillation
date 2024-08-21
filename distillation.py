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


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def prepare_dataloader():
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    return dataloader


def distill():

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=FLAGS.logdir)

    # Load pretrained teacher model
    teacher_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    teacher_model.load_state_dict(ckpt['ema_model'])

    # Initialize student model
    student_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)

    optim = torch.optim.Adam(student_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    trainer = GaussianDiffusion_distillation_Trainer(
        teacher_model, student_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    student_sampler = GaussianDiffusionSampler(
        student_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x_T = torch.randn(FLAGS.batch_size, 3, FLAGS.img_size, FLAGS.img_size).to(device)
            # Calculate distillation loss
            loss = trainer(x_T)

            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()

            # Logging
            writer.add_scalar('distill_loss', loss.item(), step)
            pbar.set_postfix(distill_loss='%.3f' % loss.item())

            # Sample and save student outputs
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                student_model.eval()
                with torch.no_grad():
                    student_samples = student_sampler(x_T)
                    grid = (make_grid(student_samples) + 1) / 2
                    
                    # Create the directory if it doesn't exist
                    sample_dir = os.path.join(FLAGS.logdir, 'sample')
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    path = os.path.join(sample_dir, 'student_%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('student_sample', grid, step)
                student_model.train()

            # Save student model
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'student_model': student_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'student_ckpt.pt'))

            # Evaluate student model
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                student_IS, student_FID, _ = evaluate(student_sampler, student_model)
                metrics = {
                    'Student_IS': student_IS[0],
                    'Student_IS_std': student_IS[1],
                    'Student_FID': student_FID,
                }
                pbar.write(
                    "%d/%d " % (step, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'student_eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")

    writer.close()


def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.distill:
        distill()
    if not FLAGS.train and not FLAGS.eval and not FLAGS.distill:
        print('Add --train, --eval and/or --distill to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
