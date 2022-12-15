
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import colors
args =sys.argv
print(args)
assert len(args)==5 #Inputs are this current filename, x array, y array and the filename
current_path,x_filename,y_filename,z_filename,filename = args
x=np.load(x_filename);y=np.load(y_filename);z=np.load(z_filename)

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    # the scatter plot:
    sc = ax.scatter(x, y,s=1,c=z,norm=colors.LogNorm())
    plt.colorbar(sc)
    # now determine nice limits by hand:
    bins = np.arange(0, 1, 0.1)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')
    ax_histx.set_yscale('log') 
    ax_histy.set_xscale('log')
    ax.set_xlabel('PL')
    ax.set_ylabel('PD')

def save_marginal_plot(x,y,filename):
    fig = plt.figure(figsize=(6, 6))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    # Draw the scatter plot and marginals.
    scatter_hist(x, y, ax, ax_histx, ax_histy)
    plt.savefig(filename+'.png')

save_marginal_plot(x,y,filename)