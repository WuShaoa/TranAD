import pandas as pd  
import os
import pywt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
  
dataset_folder = 'data/addr1394'

df_dst = pd.read_csv(os.path.join(dataset_folder,"channels_1394_DST.csv"))
df_id = pd.read_csv(os.path.join(dataset_folder,"channels_1394_ID.csv"))   
df_comb = pd.concat([df_dst, df_id], axis=1)
dd = df_comb.apply(lambda x: x.astype(str).map(lambda x: int(x, base=16)))

# 尝试不同类型的小波
wavelet_types = ['db1', 'db2', 'db3', 'db4', 'sym2', 'sym3', 'sym4', 'coif1', 'coif2', 'coif3']

data = dd.iloc[:, 0].values[:2000]
## plt.figure(figsize=(12, 8))
## for i, wavelet in enumerate(wavelet_types, start=1):
##     coeffs = pywt.wavedec(data, wavelet, level=3)
##     reconstructed_signal = pywt.waverec(coeffs, wavelet)
 #   
##     plt.subplot(5, 2, i)
##     plt.plot(data, alpha=0.5)
##     plt.plot(reconstructed_signal-data)
##     plt.title(f'Wavelet: {wavelet}')

# plt.tight_layout()
# plt.show()

# 创建一个示例信号
# 生成一个示例信号，假设长度为2000
# t = np.linspace(0, 10, num=2000)  # 时间轴
# data = np.sin(t)  # 生成正弦波信号
data = dd.iloc[:, 1].values[:2000]
t = np.linspace(0, 1, num=len(data))
scaler = StandardScaler() #MinMaxScaler()
signal = scaler.fit_transform(data.reshape(-1, 1))

# # 创建一个示例信号
# t = np.linspace(0, 1, num=len(data))
# signal = data

# # 执行小波变换
# wavelet_type = 'db4'  # 选择小波类型
# level = 4  # 分解的层数

# coeffs = pywt.wavedec(signal, wavelet_type, level=level)

# # 绘制原始信号和分解出的不同尺度的小波系数
# plt.figure(figsize=(12, 6))

# # 绘制原始信号
# plt.subplot(level + 2, 1, 1)
# plt.plot(t, signal, label='Original Signal')
# plt.legend()

# # 绘制各层小波系数
# for i in range(level):
#     plt.subplot(level + 2, 1, i + 2)
#     plt.plot(coeffs[i], label=f'Detail Coefficients {i+1}')
#     plt.legend()

# # 绘制最后一层的近似系数
# plt.subplot(level + 2, 1, level + 2)
# plt.plot(coeffs[level], label=f'Approximation Coefficients {level+1}')
# plt.legend()

# plt.tight_layout()
# plt.show()
# 执行小波包变换
wavelet_type = 'db1'#'db4'  # 选择小波类型
level = 4  # 分解的层数

wp = pywt.WaveletPacket(data=signal.squeeze(), wavelet=wavelet_type, mode='symmetric', maxlevel=level)
nodes = [node.path for node in wp.get_level(level, 'freq')]

# 绘制原始信号和小波包分解的不同子带系数
plt.figure(figsize=(12, 6))
plt.subplot(level + 2, 1, 1)
plt.plot(t, signal, label='Original Signal')
plt.legend()

for i, node in enumerate(nodes):
    coeffs = wp[node].data
    if i >= level + 1:  # 检查节点数量是否超出范围
        break
    plt.subplot(level + 2, 1, i + 2)
    plt.plot(coeffs, label=f'Node {i+1} Coefficients')
    plt.legend()

plt.tight_layout()
plt.show()
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import pywt
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

# # seq_length*batch_size*input_dim
# # 生成示例数据
# np.random.seed(42)
# data = np.random.randn(1000, 100)  # 假设有1000个时间序列，每个序列有100个数据点

# # 对数据进行标准化
# scaler = MinMaxScaler() #StandardScaler()
# data = scaler.fit_transform(data)

# # 对数据进行小波变换
# wavelet_type = 'db4'
# level = 3

# coeffs_list = []
# for i in range(len(data)):
#     coeffs = pywt.wavedec(data[i], wavelet_type, level=level)
#     coeffs_list.append(np.concatenate(coeffs))

# # 将小波系数转换为PyTorch张量
# wavelet_tensors = torch.tensor(coeffs_list, dtype=torch.float64)

# # 定义神经网络
# class Encoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )

#     def forward(self, x):
#         return self.encoder(x)

# # 初始化编码器和队列
# input_dim = wavelet_tensors.size(1)
# hidden_dim = 64
# encoder = Encoder(input_dim, hidden_dim)
# queue = torch.randn(hidden_dim, len(wavelet_tensors)).cpu()

# # 损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(encoder.parameters(), lr=0.001)

# # 数据加载器
# batch_size = 32
# data_loader = DataLoader(TensorDataset(wavelet_tensors), batch_size=batch_size, shuffle=False)

# # 训练编码器
# num_epochs = 20
# temperature = 0.07
# for epoch in range(num_epochs):
#     for batch_data in data_loader:
#         optimizer.zero_grad()

#         data_batch = batch_data[0].cpu()
#         z = encoder(data_batch)

#         logits = torch.matmul(z, queue).detach()
#         logits /= temperature
#         loss = criterion(logits, torch.diag(logits))

#         loss.backward()
#         optimizer.step()
        
#         # 打印当前损失值
#         print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item():.4f}")
#         with torch.no_grad():
#             queue[:, batch_size:] = queue[:, :-batch_size].clone()
#             queue[:, :batch_size] = z.t()

# # 提取表示
# encoder.eval()
# with torch.no_grad():
#     representations = encoder(wavelet_tensors.cpu()).cpu().numpy()

# 在这里进行异常检测，利用 representations 进行进一步的异常检测处理
# 例如，可以使用聚类、密度估计、阈值等方法进行异常检测
# ...
