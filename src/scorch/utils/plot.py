import matplotlib.pyplot as plt
import torch

def plot_images(imgs_top: torch.Tensor, imgs_bottom: torch.Tensor, title=None, cmap_top='gray', cmap_bottom='gray'):
    _, axs = plt.subplots(2, imgs_top.size(0), figsize=(imgs_top.size(0) * 1.5, 3))
    for i, (top, bottom) in enumerate(zip(imgs_top, imgs_bottom)):
        img1 = top.detach().cpu().squeeze()
        img2 = bottom.detach().cpu().squeeze()

        img1_np = img1.numpy()
        img2_np = img2.numpy()

        axs[0, i].imshow(img1_np, cmap=cmap_top)
        axs[0, i].axis('off')
        axs[1, i].imshow(img2_np, cmap=cmap_top)
        axs[1, i].axis('off')

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()
