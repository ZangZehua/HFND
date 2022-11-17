import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs

from gaussian_blur import GaussianBlur
from resnet_simclr import ResNetSimCLR
from utils import accuracy


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)


# config
data_path = "/data/data/STL-10/vi"
batch_size = 100
color_jitter = transforms.ColorJitter(0.8 * 1, 0.8 * 1, 0.8 * 1, 0.2 * 1)
transforms_low = transforms.Compose([
    transforms.ToTensor()
])
transforms_mid = transforms.Compose([
    transforms.RandomResizedCrop(size=96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([color_jitter], p=0.4),
    transforms.RandomGrayscale(p=0.1),
    GaussianBlur(kernel_size=int(0.1 * 96)),
    transforms.ToTensor()
])
transforms_high = transforms.Compose([
    transforms.RandomResizedCrop(size=96, scale=(0.2, 0.6)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([color_jitter], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=int(0.2 * 96)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(data_path, transform=transforms_low)
data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=None)

model = ResNetSimCLR(base_model="resnet18", out_dim=128).cuda()
checkpoint = torch.load('./checkpoint_0240.pth.tar', map_location=device)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)


features = torch.Tensor([])
labels = torch.Tensor([])
for counter, (x_batch, y_batch) in enumerate(data_loader):
    x_batch = x_batch.float().cuda()
    features = model(x_batch).detach().cpu()
    labels = y_batch.int()

print(features.shape, labels.shape)

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(features[:50])
X_tsne_data = np.vstack((X_tsne[:50].T, labels[:50])).T
df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class'])
df_tsne.head()

plt.figure(figsize=(8, 8))
sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2')
plt.show()

# features = torch.Tensor([])
# for i in range(10):
#     print(features_list[i].shape)

# features = torch.from_numpy(features_list)
# print(features.shape)

# for i in range(10):
#     for j in range(500):
#         for k in range(1, 500):


# #构建kmeans算法
# # features_numpy = features.numpy()
# clusters = 10  #聚类的数目为3
# k_means = KMeans(init='k-means++', n_clusters=clusters, random_state=28)
# k_means.fit(features)  #训练模型
#
# #预测结果
# km_y_hat = torch.from_numpy(k_means.predict(features))
#
# print(km_y_hat.shape, labels.shape)
# print(type(km_y_hat), type(labels))
#
# acc1 = torch.where(km_y_hat == labels, 1, 0)
#
# bag_pred = torch.zeros(10)
# for a in km_y_hat:
#     bag_pred[a] += 1
#
# print(bag_pred.sum())
#
# print(sum(acc1)/5000)
# print(bag_pred)



# ##获取聚类中心点并聚类中心点进行排序
# k_means_cluster_centers = k_means.cluster_centers_#输出kmeans聚类中心点
# print ("K-Means算法聚类中心点:\ncenter=", k_means_cluster_centers)
# ## 画图
# plt.figure(figsize=(12, 6), facecolor='w')
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)
# cm = mpl.colors.ListedColormap(['#FFC2CC', '#C2FFCC', '#CCC2FF'])
# cm2 = mpl.colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# #子图1：原始数据
# plt.scatter(features_numpy[:, 0], features_numpy[:, 1], c=labels, s=6, cmap=cm, edgecolors='none')
# plt.title(u'original data')
# plt.xticks(())
# plt.yticks(())
# plt.grid(True)
# plt.show()
#
# #子图2：K-Means算法聚类结果图
# plt.scatter(features_numpy[:,0], features_numpy[:,1], c=km_y_hat, s=6, cmap=cm,edgecolors='none')
# plt.scatter(k_means_cluster_centers[:,0], k_means_cluster_centers[:,1],c=range(clusters),s=60,cmap=cm2,edgecolors='none')
# plt.title(u'K-Means')
# plt.xticks(())
# plt.yticks(())
# plt.grid(True)
# plt.show()



