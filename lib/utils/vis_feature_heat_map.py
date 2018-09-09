import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os

import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def create_heat_maps(blob_real, blob_fake, blob_residual, pos=False):
    """
    create_heat_map: create heat map of feature map by linear combination
    :param blob_real:
    :param blob_fake:
    :param blob_residual:
    :param pos:
    :return: map_real, map_fake, map_residual
    """
    if pos:
        map_real = np.maximum(np.average(blob_real, axis=0), 0)
        map_fake = np.maximum(np.average(blob_fake, axis=0), 0)
        map_residual = np.maximum(np.average(blob_residual, axis=0), 0)
    else:
        map_real = np.average(blob_real, axis=0)
        map_fake = np.average(blob_fake, axis=0)
        map_residual = np.average(blob_residual, axis=0)

    min_real = np.min(map_real)
    min_fake = np.min(map_fake)
    # min_residual = np.min(map_residual)

    max_real = np.max(map_real)
    max_fake = np.max(map_fake)
    # max_residual = np.max(map_residual)

    # Normalize between 0-1
    map_real = (map_real - min_real) / (max_real - min_real)
    map_fake = (map_fake - min_fake) / (max_fake - min_fake)
    map_residual = (map_residual - min_fake) / (max_fake - min_fake)

    return map_real, map_fake, map_residual


def show_heat_maps(blob_real, blob_fake, blob_residual, output_dir, im_name, blob_image=None,
                   ext="jpg", pos=False):

    batch_size = blob_real.shape[0]

    for batch in range(batch_size):
        fig = plt.figure(frameon=False)

        map_real, map_fake, map_residual = create_heat_maps(blob_real[batch, :, :, :],
                                                            blob_fake[batch, :, :, :],
                                                            blob_residual[batch, :, :, :],
                                                            pos=pos
                                                            )

        if blob_image is not None:
            plt.subplot(1, 4, 1)
            try:
                plt.imshow(blob_image[batch])
                plt.show()
            except ValueError:
                pass
            plt.title('RoI from Image')

            plt.subplot(1, 4, 2)
            plt.imshow(map_real, cmap='jet', interpolation='bilinear')
            plt.axis('off')
            plt.show()
            plt.title('Real RoI')

            plt.subplot(1, 4, 3)
            plt.imshow(map_fake, cmap='jet', interpolation='bilinear')
            plt.axis('off')
            plt.show()
            plt.title('Fake RoI')

            plt.subplot(1, 4, 4)
            plt.imshow(map_residual, cmap='jet', interpolation='bilinear')
            plt.axis('off')
            plt.show()
            plt.title('Residual RoI')

        else:
            plt.subplot(1, 3, 1)
            plt.imshow(map_real, cmap='jet', interpolation='bilinear')
            plt.axis('off')
            plt.show()
            plt.title('Real RoI')

            plt.subplot(1, 3, 2)
            plt.imshow(map_fake, cmap='jet', interpolation='bilinear')
            plt.axis('off')
            plt.show()
            plt.title('Fake RoI')

            plt.subplot(1, 3, 3)
            plt.imshow(map_residual, cmap='jet', interpolation='bilinear')
            plt.axis('off')
            plt.show()
            plt.title('Residual RoI')

        output_name = os.path.basename(im_name) + '_batch_{}.'.format(batch) + ext
        fig.savefig(os.path.join(output_dir, '{}'.format(output_name)))
        plt.close('all')