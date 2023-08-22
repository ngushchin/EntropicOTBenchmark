import matplotlib
from matplotlib import pyplot as plt
import torch
import numpy as np

from sklearn.datasets import make_moons
from sklearn.decomposition import PCA


class GaussianSampler:
    def __init__(self):
        pass
    
    def sample(self, n_samples):
        return torch.randn(n_samples, 2).cuda()
    
    
class MoonsSampler:
    def __init__(self):
        pass
    
    def sample(self, n_samples):
        return (torch.tensor(make_moons(
            n_samples=n_samples, noise=0.05
        )[0].astype('float32') / 1 - np.array([0.5, 0.25])[None, :])*1.5).cuda()
    
    
def plot_mapping(independent_mapping, true_mapping, predicted_mapping, target_data, n_plot, step):
    s=30
    linewidth=0.2
    map_alpha=1
    data_alpha=1
    figsize=(5, 5)
    dpi=None
    data_color='red'
    mapped_data_color='blue'
    map_color='green'
    map_label=None
    data_label=None
    mapped_data_label=None
    
    dim = target_data.shape[-1]
    
    independent_mapping_pca = np.concatenate((        
        independent_mapping[:n_plot, :dim],
        independent_mapping[:n_plot, dim:],
        ), axis=-1)
 
    true_mapping_pca = np.concatenate((
        true_mapping[:n_plot, :dim],
        true_mapping[:n_plot, dim:],
    ), axis=-1)
    
    predicted_mapping_pca = np.concatenate((
        predicted_mapping[:n_plot, :dim],
        predicted_mapping[:n_plot, dim:],
    ), axis=-1)
    
    target_data_pca = target_data
    
    fig, axes = plt.subplots(1, 3, figsize=(15,5),squeeze=True,sharex=True,sharey=True, dpi=300)
    titles = ["independent", "true", "predicted"]
    for i, mapping in enumerate([independent_mapping_pca, true_mapping_pca, predicted_mapping_pca]):
        inp = mapping[:, :2]
        out = mapping[:, 2:]

        lines = np.concatenate([inp, out], axis=-1).reshape(-1, 2, 2)
        lc = matplotlib.collections.LineCollection(
            lines, color=map_color, linewidths=linewidth, alpha=map_alpha, label=map_label)
        axes[i].add_collection(lc)

        axes[i].scatter(
            inp[:, 0], inp[:, 1], s=s, label=data_label,
            alpha=data_alpha, zorder=2, color=data_color)
        axes[i].scatter(
            out[:, 0], out[:, 1], s=s, label=mapped_data_label,
            alpha=data_alpha, zorder=2, color=mapped_data_color)

        axes[i].scatter(target_data_pca[:1000,0], target_data_pca[:1000,1], c="orange", edgecolor = 'black',
                    label = r'$x\sim P_1(x)$', s =10)
        axes[i].grid()
        axes[i].set_title(titles[i])
        
    
def plot_two_distirubtions(samples_1, samples_2, label_1, label_2, color_1, color_2, save_img_name, ):
#     x_min, x_max = min(x_sample[:, 0].min(), y_sample[:, 0].min()), max(x_sample[:, 0].max(), y_sample[:, 0].max())
#     y_min, y_max = min(x_sample[:, 1].min(), y_sample[:, 1].min()), max(x_sample[:, 1].max(), y_sample[:, 1].max())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=300)

    axes[0].scatter(samples_1[:, 0], samples_1[:, 1], c=color_1, s=20, edgecolors="black")
    axes[0].grid()
    axes[0].set_xlim([-3, 3])
    axes[0].set_ylim([-3, 3])
    axes[0].set_title(label_1, fontsize=16)

    axes[1].scatter(samples_2[:, 0], samples_2[:, 1], c=color_2, s=20, edgecolors="black")
    axes[1].grid()
    axes[1].set_xlim([-3, 3])
    axes[1].set_ylim([-3, 3])
    axes[1].set_title(label_2, fontsize=16)
    
    plt.savefig(save_img_name)
    plt.show()
    
    
    
def plot_2d_benchmark(input_samples, target_samples, benchmark_target, mapping, save_img_name):
#     x_min, x_max = min(x_sample[:, 0].min(), y_sample[:, 0].min()), max(x_sample[:, 0].max(), y_sample[:, 0].max())
#     y_min, y_max = min(x_sample[:, 1].min(), y_sample[:, 1].min()), max(x_sample[:, 1].max(), y_sample[:, 1].max())

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.25), dpi=300)

    axes[0].scatter(input_samples[:, 0], input_samples[:, 1], c="gold", s=40, edgecolors="black")
    axes[0].grid()
    axes[0].set_xlim([-3, 3])
    axes[0].set_ylim([-3, 3])
    axes[0].set_title(r"Input distribution $\mathbb{P}_0$", fontsize=12)

    axes[1].scatter(target_samples[:, 0], target_samples[:, 1], c="white", s=40, edgecolors="black")
    axes[1].grid()
    axes[1].set_xlim([-3, 3])
    axes[1].set_ylim([-3, 3])
    axes[1].set_title(r"Real target distribution $\mathbb{P}_1$", fontsize=12)
    
    axes[2].scatter(benchmark_target[:, 0], benchmark_target[:, 1], c="lightgrey", s=40, edgecolors="black")
    axes[2].grid()
    axes[2].set_xlim([-3, 3])
    axes[2].set_ylim([-3, 3])
    axes[2].set_title(r"Benchmark target distribution $\widehat{\mathbb{P}}_1$", fontsize=12)
    
    axes[3].scatter(target_samples[:, 0], target_samples[:, 1], c="white", s=40, edgecolors="black", alpha=1)
    axes[3].grid()
    axes[3].set_xlim([-3, 3])
    axes[3].set_ylim([-3, 3])
    
    n_x, n_samples = mapping.shape[:2]
    lines = mapping.reshape(n_x*n_samples, 2, 2)
    lc = matplotlib.collections.LineCollection(
        lines, color="green", linewidths=2, alpha=1, label=None)
    axes[3].add_collection(lc)

    mapping = mapping.reshape(n_x, n_samples, 4)
    axes[3].scatter(
        mapping[:, 0, 0], mapping[:, 0, 1], s=40, label=None,
        alpha=1, zorder=2, color="red")
    axes[3].scatter(
        mapping[:, :, 2], mapping[:, :, 3], s=40, label=None,
        alpha=1, zorder=2, color="blue")
    axes[3].set_title(r"Benchmark EOT plan $\pi^*(\cdot| x)$", fontsize=12)
    fig.tight_layout(pad=0.001)
    plt.savefig(save_img_name)
    plt.show()
#     =r"Input distribution $\mathbb{P}_0$", color_1="gold",
    
#                        label_1=r"Real target distribution $\mathbb{P}_1$", color_1="white",
#                        label_2=r"Constructed benchmark target distribution $\widehat{\mathbb{P}}_1$"
# plt.show()