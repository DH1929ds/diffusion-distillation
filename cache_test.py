# import numpy as np
# import matplotlib.pyplot as plt
# import imageio

# # 초기 설정: 128개의 개체를 1000 스텝으로 시작
# num_entities = 128
# initial_steps = 1000
# entities = np.full(num_entities, initial_steps)  # 각 개체의 초기 스텝은 1000으로 설정
# iterations = 10000  # GIF의 프레임 수를 제어하기 위해 100번 반복
# select_count = 512  # 선택할 개체의 수

# # GIF 생성을 위한 리스트
# frames = []

# # 반복 수행
# for i in range(iterations):
#     # 1000 스텝의 개체가 128로 일정하게 유지되도록 보충
#     count_1000_steps = np.sum(entities == initial_steps)
#     if count_1000_steps < num_entities:
#         entities = np.append(entities, [initial_steps] * (num_entities - count_1000_steps))
    
#     # 확률 분포 계산: 각 개체의 스텝 값에 비례하도록 설정
#     probabilities = entities / np.sum(entities)
    
#     # 개체 선택 (비복원 추출)
#     selected_indices = np.random.choice(len(entities), size=select_count, p=probabilities)
    
#     # 선택된 개체들의 스텝 감소
#     entities[selected_indices] -= 1
    
#     # 스텝이 0에 도달한 개체 제거
#     entities = entities[entities > 0]
    
#     # 히스토그램 생성
#     plt.figure(figsize=(10, 6))
#     plt.hist(entities, bins=1000, range=(0, initial_steps), color='blue', alpha=0.7)
#     plt.title(f'Iteration {i+1}: Step Distribution')
#     plt.xlabel('Steps')
#     plt.ylabel('Number of Entities')
    
#     # 히스토그램을 이미지로 저장
#     plt_filename = f'/home/dohyun/kdh/diffusion_distillation/cache_test/{i:03d}.png'
#     plt.savefig(plt_filename)
#     plt.close()
#     frames.append(plt_filename)

# # GIF 생성
# gif_filename = '/home/dohyun/kdh/diffusion_distillation/step_distribution.gif'
# with imageio.get_writer(gif_filename, mode='I', duration=0.1) as writer:
#     for frame in frames:
#         image = imageio.v2.imread(frame)
#         writer.append_data(image)

# gif_filename


# import numpy as np
# import matplotlib.pyplot as plt
# import imageio

# # 초기 설정: 128개의 개체를 1000 스텝으로 시작
# num_entities = 128
# initial_steps = 1000
# entities = np.full(num_entities, initial_steps)  # 각 개체의 초기 스텝은 1000으로 설정
# iterations = 10000  # GIF의 프레임 수를 제어하기 위해 100번 반복
# select_count = 512  # 선택할 개체의 수

# # GIF 생성을 위한 리스트
# frames = []

# # 반복 수행
# for i in range(iterations):
#     # 1000 스텝의 개체가 128로 일정하게 유지되도록 보충
#     count_1000_steps = np.sum(entities == initial_steps)
#     if count_1000_steps < num_entities:
#         entities = np.append(entities, [initial_steps] * (num_entities - count_1000_steps))
    
#     # 확률 분포 계산: 각 개체의 스텝 값에 비례하도록 설정
#     probabilities = entities / np.sum(entities)
    
#     # 개체 선택 (비복원 추출)
#     selected_indices = np.random.choice(len(entities), size=select_count, p=probabilities)
    
#     # 선택된 개체들의 스텝 감소
#     entities[selected_indices] -= 1
    
#     # 스텝이 0에 도달한 개체 제거
#     entities = entities[entities > 0]
    
#     # 히스토그램 생성
#     plt.figure(figsize=(10, 6))
#     plt.hist(entities, bins=1000, range=(0, initial_steps), color='blue', alpha=0.7)
#     plt.title(f'Iteration {i+1}: Step Distribution')
#     plt.xlabel('Steps')
#     plt.ylabel('Number of Entities')
    
#     # 히스토그램을 이미지로 저장
#     plt_filename = f'/home/dohyun/kdh/diffusion_distillation/cache_test/{i:03d}.png'
#     plt.savefig(plt_filename)
#     plt.close()
#     frames.append(plt_filename)

# # GIF 생성
# gif_filename = '/home/dohyun/kdh/diffusion_distillation/step_distribution.gif'
# with imageio.get_writer(gif_filename, mode='I', duration=0.1) as writer:
#     for frame in frames:
#         image = imageio.v2.imread(frame)
#         writer.append_data(image)

# gif_filename


# import numpy as np
# import matplotlib.pyplot as plt
# import imageio
# import time

# # 초기 설정: 128개의 개체를 1000 스텝으로 시작
# initial_steps = 1000
# entities = np.zeros(initial_steps)  # 각 개체의 초기 스텝은 1000으로 설정
# select = np.zeros(initial_steps)
# iterations = 1000000  # GIF의 프레임 수를 제어하기 위해 10000번 반복
# select_count = 512  # 선택할 개체의 수

# # GIF 생성을 위한 리스트
# frames = []

# # 반복 수행
# for i in range(iterations):
#     start_time = time.time()

#     entities[0] = select_count

#     non_zero_indices = np.where(entities > 0)[0]

#     # 확률 계산 시 entities[0]을 4로 간주하여 확률을 계산
#     adjusted_entities = entities.copy()
#     adjusted_entities[0] = 10  # entities[0]을 4로 간주
#     probabilities = adjusted_entities[non_zero_indices] / adjusted_entities[non_zero_indices].sum()

#     # 선택할 개체 수를 각 entities 값에 의해 제한
#     max_selectable = entities[non_zero_indices].astype(int)

#     # 선택할 개체의 인덱스를 추적
#     selected_indices = []

#     # 반복적으로 개체를 확률에 따라 선택
#     while len(selected_indices) < select_count:
#         idx = np.random.choice(non_zero_indices, p=probabilities)
        
#         # 선택 횟수가 개체의 수보다 적으면 추가
#         if selected_indices.count(idx) < max_selectable[non_zero_indices.tolist().index(idx)]:
#             selected_indices.append(idx)

#     # selected_indices의 빈도 수를 select에 반영
#     select = np.bincount(selected_indices, minlength=initial_steps)

#     entities -= select
#     entities[1:] += select[:999]

#     end_time = time.time()

#     print(np.sum(entities))
#     elapsed_time = end_time - start_time
#     print(f"Iteration {i+1}: Time elapsed = {elapsed_time:.6f} seconds")
#     # print(entities)

#     if i % 30 ==0:
#         # 결과 시각화를 위해 프레임 저장
#         plt.figure(figsize=(6, 4))
#         plt.bar(range(len(entities)), entities)
#         plt.title(f"Iteration {i+1}")
#         plt.xlabel("Entities")
#         plt.ylabel("Steps")
#         plt.ylim(0, 550)

#         # 현재 플롯을 이미지로 변환하여 프레임에 추가
#         plt.savefig('/home/dohyun/kdh/diffusion_distillation/cache_test/temp_frame.png')
#         plt.close()
#         frames.append(imageio.imread('/home/dohyun/kdh/diffusion_distillation/cache_test/temp_frame.png'))

# # 생성된 프레임으로 GIF 저장
# imageio.mimsave('/home/dohyun/kdh/diffusion_distillation/cache_test/output.gif', frames, fps=30)


import numpy as np
import matplotlib.pyplot as plt
import imageio

iterations = 1000000 
t_cache = np.ones(5000, dtype=np.int32) * 999
frames = []

for i in range(iterations):
    indices = np.random.permutation(t_cache.size)

    t_cache = t_cache[indices]

    t = t_cache[:128]

    ### cache update ###
    t_cache[:128] -= 1
    
    num_999 = np.sum(t_cache == 999)
    if num_999 < 5:
        # 부족한 개수만큼 999로 설정
        missing_999 = 5 - num_999
        zero_indices = np.where(t_cache != 999)[0]
        t_cache[zero_indices[:missing_999]] = 999
    
    # t_cache에서 값이 0인 인덱스를 찾아 초기화
    zero_indices = np.where(t_cache == 0)[0]
    num_zero_indices = zero_indices.size

    # 0인 인덱스가 있는 경우에만 초기화 수행
    if num_zero_indices > 0:
        t_cache[zero_indices] = np.random.randint(1, 1000, size=num_zero_indices, dtype=np.int32)


    # 100번의 반복마다 히스토그램 생성
    if (i + 1) % 100 == 0:
        plt.figure(figsize=(10, 6))
        plt.hist(t_cache, bins=1000, range=(0, 1000), color='blue', alpha=0.7)
        plt.title(f'Iteration {i+1}: Step Distribution')
        plt.xlabel('Steps')
        plt.ylabel('Number of Entities')
        
        plt.savefig('/home/dohyun/kdh/diffusion_distillation/cache_test/temp_frame.png')
        plt.close()
        frames.append(imageio.imread('/home/dohyun/kdh/diffusion_distillation/cache_test/temp_frame.png'))

# GIF 생성
gif_filename = '/home/dohyun/kdh/diffusion_distillation/step_distribution.gif'
with imageio.get_writer(gif_filename, mode='I', duration=0.1) as writer:
    for frame in frames:
        writer.append_data(frame)

gif_filename
