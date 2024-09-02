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
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from diffusion import GaussianDiffusion_distillation_Trainer, GaussianDiffusionTrainer, GaussianDiffusionSampler, GaussianDiffusion
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
    save_dir = '/home/dohyun/kdh/diffusion_distillation/t-sne_result'
    # Load pretrained UNet model
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    model.load_state_dict(ckpt['ema_model'])
    model.eval()

    diffusion = GaussianDiffusion(
        FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)

    # Load CIFAR-10 dataset and randomly select 128 samples
    dataset = CIFAR10(
        root='./data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    #############################################################################################################

    # # 10개의 랜덤 샘플 선택
    # indices = random.sample(range(len(dataset)), 5)
    # samples = torch.stack([dataset[i][0] for i in indices]).to(device)

    # # 강제로 서로 다른 레이블(0~9) 할당
    # sample_labels = list(range(10))
    
    ##############################################################################################################
    # CIFAR-10 데이터셋에서 클래스별 인덱스 수집
    class_indices = {i: [] for i in range(10)}  # CIFAR-10은 10개의 클래스(0~9)를 가짐

    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # 각 클래스에서 10개의 샘플을 랜덤으로 선택
    selected_indices = []
    for class_id, indices in class_indices.items():
        selected_indices.extend(random.sample(indices, 2))

    # 선택된 100개의 샘플을 가져옴
    samples = torch.stack([dataset[i][0] for i in selected_indices]).to(device)

    # 선택된 100개의 샘플에 해당하는 레이블 가져오기
    sample_labels = [dataset[i][1] for i in selected_indices]
    
    ##############################################################################################################

    # Feature 추출을 위한 저장소
    all_features = []
    all_labels = []  # 레이블 저장 (샘플, 노이즈 타임스텝, 모델 타임스텝)
    
    # 128개의 샘플에 대해
    for noise_timestep in tqdm(range(0, FLAGS.T,10), desc="Noise Timesteps Progress"):
        for model_timestep in range(0, FLAGS.T, 100):
            # 현재 timestep에서 128개 샘플에 대해 노이즈 추가
            t_noise = torch.ones(samples.size(0), device=samples.device, dtype=torch.long) * noise_timestep
            noised_samples = diffusion.diffusion(samples, t_noise)

            t_model = torch.ones(samples.size(0), device=samples.device, dtype=torch.long) * model_timestep
            # 1000개의 서로 다른 모델 입력 timestep에 대해 feature 추출
            # for model_timestep in range(0, FLAGS.T, 100):
            #     t_model = torch.ones(samples.size(0), device=samples.device, dtype=torch.long) * model_timestep
                
                # 모델을 사용하여 feature 추출
            with torch.no_grad():
                outputs = model.extract_feature(noised_samples, t_noise)
                
            # # Feature size 출력
            # print(f"Feature size at noise_timestep {noise_timestep}, model_timestep {model_timestep}: {outputs.size()}")
            
            # Flatten the outputs for t-SNE
            outputs_flat = outputs.view(outputs.size(0), -1).cpu().numpy()
            all_features.append(outputs_flat)
            
            # 레이블 추가
            all_labels.extend([(sample_labels[i], noise_timestep, model_timestep) for i in range(samples.size(0))])

    # 모든 feature를 (128*1000*1000, feature_size)의 형태로 결합
    all_features_concat = np.concatenate(all_features, axis=0)
    
    # 레이블 분리
    sample_labels_list = [label[0] for label in all_labels]  # 샘플 레이블
    noise_timesteps_list = [label[1] for label in all_labels]  # 노이즈 타임스텝 레이블
    model_timesteps_list = [label[2] for label in all_labels]  # 모델 타임스텝 레이블

    print('start t-sne')
    ###############################################################################
    # # Apply t-SNE to the outputs
    # tsne = TSNE(n_components=2, random_state=42)
    # tsne_results = tsne.fit_transform(all_features_concat)

    # # 1. Sample 레이블에 따른 t-SNE 시각화
    # plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=sample_labels_list, s=10, cmap='tab10')
    # plt.title('t-SNE of UNet Outputs on CIFAR-10 Samples (Sample Labels)')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # plt.colorbar(scatter, ticks=range(10), label='CIFAR-10 Classes')
    # plt.savefig(os.path.join(save_dir, 't-SNE_sample_labels.png'))  # 결과 저장
    # plt.close()

    # # 2. Noise 타임스텝 레이블에 따른 t-SNE 시각화
    # plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=noise_timesteps_list, s=10, cmap='viridis')
    # plt.title('t-SNE of UNet Outputs with Noise Timesteps')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # plt.colorbar(scatter, label='Noise Timesteps')
    # plt.savefig(os.path.join(save_dir, 't-SNE_noise_timesteps.png'))  # 결과 저장
    # plt.close()

    # # 3. Model 타임스텝 레이블에 따른 t-SNE 시각화
    # plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=model_timesteps_list, s=10, cmap='plasma')
    # plt.title('t-SNE of UNet Outputs with Model Timesteps')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # plt.colorbar(scatter, label='Model Timesteps')
    # plt.savefig(os.path.join(save_dir, 't-SNE_model_timesteps.png'))  # 결과 저장
    # plt.close()

    ###############################################################################
    # 3D로 t-SNE 적용
    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(all_features_concat)

    # 1. Sample 레이블에 따른 t-SNE 시각화 (3D)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=sample_labels_list, s=10, cmap='tab10')
    ax.set_title('3D t-SNE of UNet Outputs on CIFAR-10 Samples (Sample Labels)')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.colorbar(scatter, ticks=range(10), label='CIFAR-10 Classes')
    plt.savefig(os.path.join(save_dir, '3D_t-SNE_sample_labels.png'))  # 결과 저장
    plt.close()

    # 2. Noise 타임스텝 레이블에 따른 t-SNE 시각화 (3D)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=noise_timesteps_list, s=10, cmap='viridis')
    ax.set_title('3D t-SNE of UNet Outputs with Noise Timesteps')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.colorbar(scatter, label='Noise Timesteps')
    plt.savefig(os.path.join(save_dir, '3D_t-SNE_noise_timesteps.png'))  # 결과 저장
    plt.close()

    # 3. Model 타임스텝 레이블에 따른 t-SNE 시각화 (3D)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=model_timesteps_list, s=10, cmap='plasma')
    ax.set_title('3D t-SNE of UNet Outputs with Model Timesteps')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.colorbar(scatter, label='Model Timesteps')
    plt.savefig(os.path.join(save_dir, '3D_t-SNE_model_timesteps.png'))  # 결과 저장
    plt.close()

    ###############################################################################


def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    t_sne()


if __name__ == '__main__':
    app.run(main)
