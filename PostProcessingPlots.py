# ---------------------------------------------------------------------------
# --- This code performs the following tasks --
# Generate various plots
# ------------------
# Authors: Doruk Aksoy; University of California, Irvine
# Contact: (daksoy@uci.edu)
# Date: 06/23/2023
# ------------------
# Version: Python 3.8
# Execution: python3 Plots.py
# ---------------------------------------------------------------------------
# %% Imports
import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.color import label2rgb
# %% Helper Methods
def plot_multiple_images_single_row(images_list, titles=['Figure'], fig_size=(5,6), cmap=None):
    if len(images_list)>1:
        fig, axes = plt.subplots(nrows=1, ncols=len(images_list), figsize=(len(images_list)*5, 6),
                                 sharex=True, sharey=True)
        ax = axes.ravel()
        for i in range(len(images_list)):
            plt.subplot(1, len(images_list), i + 1)
            if len(images_list) == len(titles): ax[i].set_title(titles[i],fontsize=25)
            # else: print('Number of titles does not match the number of figures.')
            if len(images_list[i].shape)==3:
                ax[i].imshow(tf.keras.preprocessing.image.array_to_img(images_list[i]))
            elif len(images_list[i].shape)==2:
                ax[i].imshow(images_list[i], cmap=cmap or plt.cm.gray)
            else:
                raise Exception("Incorrect shape for input images. Should be (h,w,c) or (h,w)")
            plt.axis('off')
    else:
        fig, ax = plt.subplots(figsize=fig_size)
        if len(images_list) == len(titles): ax.set_title(titles[0],fontsize=25)
        else: print('Number of titles does not match the number of figures.')
        if len(images_list[0].shape)==3:
            ax.imshow(tf.keras.preprocessing.image.array_to_img(images_list[0]))
        elif len(images_list[0].shape)==2:
            ax.imshow(images_list[0], cmap=cmap or plt.cm.gray)
        else:
            raise Exception("Incorrect shape for input images. Should be (h,w,c) or (h,w)")
        plt.axis('off')
            
    fig.tight_layout()
    plt.show()

def plot_vectors_on_image(l_junction_angles, scale=10, color='yellow'):

    for (y, x), angle in l_junction_angles.items():
        dy = scale * np.cos(math.radians(angle)) # angle in degrees
        dx = -scale * np.sin(math.radians(angle)) # this is negative, due to the orientation of the mask (y-axis is reversed)
        plt.arrow(x, y, dy, dx, head_width=2, head_length=3, color=color, alpha=0.8)

    plt.axis('off')
    plt.show()

def plot_image_with_labels(image, labels, title='Segmentation'):
    # Create a colored image using label2rgb
    image_label_overlay = label2rgb(labels, image=image)
    
    # Plot
    plot_multiple_images_single_row([image_label_overlay],[title], fig_size=(12,12))
        
def post_classified_image(classified_image, iso_points, image_ind = '', title='After post-processing (Classified)'):
    
    # Define a custom colormap
    from matplotlib.colors import ListedColormap
    color_list = ['black', 'gray', 'cyan', 'red', 'yellow', 'green', 'blue', 'purple'][:len(np.unique(classified_image))]
    cmap = ListedColormap(color_list)
    
    # Plot results   
    plot_multiple_images_single_row([classified_image], fig_size=(10,10), cmap=cmap, titles=[title])
    # Visualize isolated points and angles (add to the previous figure as yellow and green arrows)
    plot_vectors_on_image(iso_points, scale=4, color='green')

def segmentation_plot(mask, mask_labels, pred_before, pred_before_labels, pred_after, pred_after_labels, image_ind = ''):
    # Plot the segmentation maps with labels
    plot_image_with_labels(mask, mask_labels, "FSGT")
    plot_image_with_labels(pred_before, pred_before_labels, "Before Post-processing")
    plot_image_with_labels(pred_after, pred_after_labels, "After Post-processing")
    
# Helper function to plot a segment
def plot_segment(segment, ax, color='r', segment_type='line', index=None, font_size=20, line_width=1, font_color='white'):
    start, end = segment[0], segment[-1]
    
    if segment_type == 'broken':
        # Plotting the points on the segment
        x_vals = [point[1] for point in segment]
        y_vals = [point[0] for point in segment]
        ax.scatter(x_vals, y_vals, color=color, s=8)
    else:
        # Plotting a line between the start and end of the segment
        ax.plot([start[1], end[1]], [start[0], end[0]], color=color, linewidth=line_width)
    
    # Displaying the index at the midpoint of the segment
    if index is not None:
        mid_idx = len(segment) // 2
        midpoint = segment[mid_idx]
        ax.text(midpoint[1], midpoint[0], str(index), fontsize=font_size, color=font_color, ha='center', va='center')
    
    plt.axis('off')

def plot_multiple_segments_on_image(labeled_elements, segment_dict, title="", font_size=20, line_width=1, font_color=None, crop_x=None, crop_y=None):
    """
    Plot multiple segments on the labeled_elements.
    segment_dict: dictionary containing the type of segments (e.g., 'broken', 'completion') as keys and their respective segments and colors as values.
    """
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(14, 14))
    labeled_elements = np.where(labeled_elements > 1, 1, labeled_elements)
    ax.imshow(labeled_elements, cmap=ListedColormap(['black', 'white']), interpolation="lanczos")
    
    for segment_type, segment_info in segment_dict.items():
        segments = segment_info['segments']
        color = segment_info['color']
        for index, segment in segments.items():
            if font_color:
                plot_segment(segment, ax, color=color, segment_type=segment_type, index=index, font_size=font_size, line_width=line_width, font_color=font_color)
            else:
                plot_segment(segment, ax, color=color, segment_type=segment_type, line_width=line_width, font_color=font_color)
    
    if crop_x is not None: ax.set_xlim(crop_x)
    if crop_y is not None: ax.set_ylim(crop_y)
    
    ax.set_title(title)
    plt.axis('off')
    plt.show()