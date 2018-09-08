import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os

import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def create_heat_map(blob):
    """
    create_heat_map: create heat map of feature map by linear combination
    :param blob:
    :return:
    """
    map = np.maximum(np.average(blob, axis=0), 0)
    map = (map - np.min(map)) / (np.max(map) - np.min(map))  # Normalize between 0-1
    map = np.uint8(map * 255)
    return map


def show_heat_maps(blob_real, blob_fake, blob_residual, output_dir, im_name, blob_image=None, ext="jpg"):
    batch_size = blob_real.shape[0]

    for batch in range(batch_size):
        fig = plt.figure(frameon=False)

        if blob_image is not None:
            plt.subplot(1, 4, 1)
            plt.imshow(blob_image[batch, :, :, :])
            plt.show()
            plt.title('RoI from Image')

            plt.subplot(1, 4, 2)
            plt.imshow(create_heat_map(blob_real[batch, :, :, :]), cmap='jet')
            plt.show()
            plt.title('Real RoI')

            plt.subplot(1, 4, 3)
            plt.imshow(create_heat_map(blob_fake[batch, :, :, :]), cmap='jet')
            plt.show()
            plt.title('Fake RoI')

            plt.subplot(1, 4, 4)
            plt.imshow(create_heat_map(blob_residual[batch, :, :, :]), cmap='jett')
            plt.show()
            plt.title('Residual RoI')

        else:
            plt.subplot(1, 3, 1)
            plt.imshow(create_heat_map(blob_real[batch, :, :, :]), cmap='jet')
            plt.show()
            plt.title('Real RoI')

            plt.subplot(1, 3, 2)
            plt.imshow(create_heat_map(blob_fake[batch, :, :, :]), cmap='jet')
            plt.show()
            plt.title('Fake RoI')

            plt.subplot(1, 3, 3)
            plt.imshow(create_heat_map(blob_residual[batch, :, :, :]), cmap='jet')
            plt.show()
            plt.title('Residual RoI')

        output_name = os.path.basename(im_name) + '_batch_{}.'.format(batch) + ext
        fig.savefig(os.path.join(output_dir, '{}'.format(output_name)))
        plt.close('all')