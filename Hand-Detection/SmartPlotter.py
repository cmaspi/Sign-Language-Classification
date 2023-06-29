import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def PLOT(images, shape, titles=None,
         colorbar=False, cmap='gray',
         seaborn_style=True, dpi=100, dark=False,
         fig_sup_title=None, axis='off',
         return_fig=False, colorbar_range=None):
    if not colorbar_range:
        colorbar_range = [[None, None]]*len(images)
    if seaborn_style:
        sns.set()
    if dark:
        plt.style.use('dark_background')

    figsize = ((5 if colorbar else 4)*shape[1], 4*shape[0])
    fig, ax = plt.subplots(*shape, figsize=figsize, dpi=dpi)
    if shape[0] == 1:
        ax = [ax]
        if shape[1] == 1:
            ax = [ax]
    x, y = shape
    for itr in range(images.__len__()):
        i, j = itr//y, itr % y
        c = ax[i][j].imshow(images[i*y+j], cmap=cmap,
                            vmin=colorbar_range[i*y+j][0],
                            vmax=colorbar_range[i*y+j][1])
        if colorbar:
            plt.colorbar(c)
        if titles:
            ax[i][j].set_title(titles[i*y+j])

    if axis != 'on':
        for i in range(x):
            for j in range(y):
                ax[i][j].axis('off')

    if fig_sup_title:
        fig.suptitle(fig_sup_title)

    plt.tight_layout()
    sns.reset_orig()
    if return_fig:
        return fig, ax
    return ax


def SmartPlot(images, shape=None, **kwargs):
    """
    Args:
       images, shape, titles, colorbar, cmap, seaborn_style,
       dpi, dark, fig_sup_title, axis, return_fig, colorbar_range
    """
    if shape is None:
        n = len(images)
        m = np.sqrt(n)
        if np.abs(m-np.round(m)) < 1e-6:
            x, y = [int(np.round(m))]*2
        elif n % 5 == 0 and n//5 <= 6:
            x, y = n//5, 5
        elif n % 4 == 0 and n//4 <= 5:
            x, y = n//4, 4
        elif n % 3 == 0 and n//3 <= 4:
            x, y = n//3, 3
        elif n % 2 == 0 and n//2 <= 3:
            x, y = n//2, 2
        else:
            x, y = int(m)+1, int(m)+1
            if x*(y-1) >= n:
                y -= 1
        shape = (x, y)
    return PLOT(images, shape, **kwargs)


class sns_style(object):
    def __init__(self, axis='off', theme=None, edge_color=False) -> None:
        self.axis = axis
        self.theme = theme
        self.edge_color = edge_color

    def __enter__(self):
        sns.set()
        if self.theme:
            plt.style.use(self.theme)
        if self.axis == 'off':
            plt.axis('off')
        plt.rcParams['Patch.force_edgecolor'] = self.edge_color

    def __exit__(self, *args):
        sns.reset_orig()
