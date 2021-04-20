import numpy as np
import matplotlib.pyplot as plt
import os
import random
import statistics
from math import sqrt
from functools import reduce
from matplotlib.pyplot import imshow
from collections import Counter

def ravel_images(images):
    raveled_images = np.array([np.ravel(image) for image in images])
    return raveled_images

def move_axis_to_batch_minor(batch_major_data, batch_axis):
    #moves batch axis to the last axis
    return np.moveaxis(batch_major_data, batch_axis, -1)

def move_axis_to_batch_major(batch_major_data, batch_axis):
    #moves batch axis to the last axis
    return np.moveaxis(batch_major_data, batch_axis, 0)

def factors(n):
    step = 2 if n%2 else 1
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(sqrt(n))+1, step) if n % i == 0)))

#def number_of_duplicates(list_a, list_b):
#    count_a = Counter(list_a)
#    count_b = Counter(list_b)
#
#    common_keys = set(count_a.keys()).intersection(count_b.keys())
#    return sum(min(count_a[key], count_b[key]) for key in common_keys)

def get_population_activities(history, threshold, freq=False):
    freq_list = np.zeros(np.shape(history)[1])
    for i in range(np.shape(history)[1]):
        count = len([j for j in history[:,i] if j > threshold])
        freq_list[i] = count
    if freq == False:
        return(freq_list/len(history))
    else:
        return(freq_list)

def get_num_unseen_points(datahistory, training_size):
    num_unseen_points_list = np.zeros(np.shape(datahistory[1]))
    prop_of_seen_data_list = np.zeros(np.shape(datahistory[1]))
    pres_length = np.shape(datahistory[0])
    seen_points = np.array([])
    for i in range(np.shape(datahistory)[1]):
        next_pres = datahistory[:,i]
#         num_of_repeat_pres = number_of_duplicates(seen_points, next_pres)
#         repeated_data_list[i] = num_of_repeat_pres
        num_unseen_points = len(np.setdiff1d(next_pres, seen_points))
        num_unseen_points_list[i] = num_unseen_points
        seen_points = np.ravel(datahistory[:,:i+1])
        prop_of_unseen_data = len(np.setdiff1d(np.arange(training_size), seen_points))/training_size
        prop_of_seen_data_list[i] = 1-prop_of_unseen_data
        if prop_of_unseen_data == 0:
            break
    return(num_unseen_points_list[:i+1], prop_of_seen_data_list[:i+1])

def get_population_activities(history, threshold, freq=False):
    freq_list = np.zeros(np.shape(history)[1])
    for i in range(np.shape(history)[1]):
        count = len([j for j in history[:,i] if j > threshold])
        freq_list[i] = count
    if freq == False:
        return(freq_list/len(history))
    else:
        return(freq_list)

def factors(n):
    step = 2 if n%2 else 1
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(sqrt(n))+1, step) if n % i == 0)))

def display_patches(n_patches,
                    imageset,
                    filestr=None,
                    figsize=(8,8),
                    patch_dims=(16,16),
                    pcobject=None,
                    center=False,
                    labels=True,
                    title=None,
                    cmap='gray',
                    acts=None,
                    sample=False):
    '''
    Function that takes in batch major data (must be a list)
    '''
    fs = factors(n_patches)
    if len(fs)%2 == 0:
        dims = [statistics.median_low(fs), statistics.median_high(fs)]
    else:
        dims = [statistics.median(fs), statistics.median(fs)]

    fig = plt.figure(figsize=(8,8))
    axs = [fig.add_subplot(dims[0],dims[1],i+1) for i in range(n_patches)]

    if labels == True:
        cols = ['{}'.format(col+1) for col in range(dims[1])]
        rows = ['{}'.format(row+1) for row in range(dims[0])]
        for ax, col in zip(np.array(axs).reshape(dims)[0], cols):
            ax.set_title(col)
        for ax, row in zip(np.array(axs).reshape(dims)[:,0], rows):
            ax.set_ylabel(row, rotation=0, size='large', labelpad=10)
    if title is not None:
        fig.suptitle(title, fontsize=18, y=1.00)

    if type(imageset) != list:
        imageset = [imageset[i,:] for i in range(imageset.shape[0])]

    if acts is not None:
        sorted_idxs = np.argsort(np.linalg.norm(acts, ord=1, axis=1))[::-1]
        imageset = [imageset[idx] for idx in sorted_idxs]

    if sample == False:
        imagelist = zip(axs, imageset)
    else:
        imagelist = zip(axs, random.sample(imageset, n_patches))

    for ax, image in imagelist:
        if center == True:
            vmax = np.max(np.abs([image.min(),image.max()]))
            vmin = -vmax
        else:
            vmax = None
            vmin = None
        if pcobject:
                image = np.reshape(pcobject.inverse_transform(image), patch_dims)
        else:
            image = np.reshape(image, patch_dims)
        ax.imshow(image, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_ticks([])

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.set_figheight(dims[0])
    fig.set_figwidth(dims[1])
    figure = plt.gcf()
    if filestr is not None:
        figure.savefig(filestr, dpi='figure')
    return(fig, axs)

def get_rf_overlap_per_weight(RF, W):
    rf_product = np.dot(RF.T, RF)
    rf_overlap = rf_product[np.triu_indices(len(rf_product),1)]
    inhibitory_wts = W[np.triu_indices(len(W), 1)]
    return(inhibitory_wts[inhibitory_wts], rf_overlap[inhibitory_wts])

def unit_plots(activities):
    fs = factors(len(activities))
    if len(fs)%2 == 0:
        dims = [statistics.median_low(fs), statistics.median_high(fs)]
    else:
        dims = [statistics.median(fs), statistics.median(fs)]
    ylim = np.max(activities)+math.floor(.2*np.max(activities))
  # f, axarr = plt.subplots(dims[0], dims[1])
    # if idxs == None:
    #     idxs = random.sample(list(range(0,len(imageset))), dims[0]*dims[1])
    # else:
    #     pass
    fig = plt.figure(figsize=(10,10))
    axs = [fig.add_subplot(dims[0],dims[1],i+1) for i in range(len(activities))]
    for ax, timecourse in zip(axs, range(len(activities))):
        ax.plot(activities[timecourse,:])
        ax.set_ylim(0,ylim)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.set_figheight(dims[0])
    fig.set_figwidth(dims[1])
    return(fig, axs)

def rf_overlap_grid_plot(Whistory, rfoverlaphistory):
    fs = factors(len(Whistory))
    if len(fs)%2 == 0:
        dims = [statistics.median_low(fs), statistics.median_high(fs)]
    else:
        dims = [statistics.median(fs), statistics.median(fs)]
#     ylim = np.max(rfoverlaphistory)+math.floor(.2*np.max(rfoverlaphistory))
  # f, axarr = plt.subplots(dims[0], dims[1])
    # if idxs == None:
    #     idxs = random.sample(list(range(0,len(imageset))), dims[0]*dims[1])
    # else:
    #     pass
    fig = plt.figure(figsize=(10,10))
    axs = [fig.add_subplot(dims[0],dims[1],i+1) for i in range(len(Whistory))]
    iteration = 0
    for ax, timecourse in zip(axs, range(len(Whistory))):
        ax.scatter(Whistory[timecourse][Whistory[timecourse]>10**-12], rfoverlaphistory[timecourse][Whistory[timecourse]>10**-12])
        ax.set_xscale('log')
        ax.set_ylim(-.5,.5)
        ax.set_xlim(10**-6, 10**2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_ticks([])
        if (iteration) % dims[1] == 0:
#             ax.text(10, .4, '{}'.format(iteration*50), style='italic')
            ax.set_ylabel('{}'.format(iteration), rotation='horizontal')
        ax.yaxis.set_label_position("left")
        ax.yaxis.set_label_coords(-.2,.5)
        iteration+=1
    fig.subplots_adjust(wspace=0, hspace=0)
    if dims[0] < dims[1]:
        fig.set_figheight(10)
        fig.set_figwidth(dims[1]/dims[0]*10)
    return(fig, axs)

def display_image(images, reshape_dims=None, center=False, idx=None):
    if idx == None:
        idx = np.random.randint(0,np.shape(images)[-1])
    if reshape_dims == None:
        data = images[:,:,idx]
    else:
        data = images[:,idx].reshape(reshape_dims)
    if center == True:
        vmax = np.max(np.abs([data.min(),data.max()]))
        vmin = -vmax
    else:
        vmax = None
        vmin = None
    plt.imshow(data, interpolation='nearest',cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()
    print('Idx: {} Mean: {} Stdev: {} Min: {} Max: {}'.format(idx, np.mean(data), np.std(data), data.min(), data.max()))
