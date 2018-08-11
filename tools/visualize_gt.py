import os
import cv2
import numpy as np
import argparse
import sys

import lib.utils.env as envu
# Matplotlib requires certain adjustments in some environments
# Must happen before importing matplotlib
envu.set_up_matplotlib()
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from core.config import cfg

plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display ground-truth images with bboxes."
    )

    parser.add_argument(
        '--json_file',
        dest='json_file',
        type=str,
        help='Include here the path of the wanted annotations file.',
        default=''
    )

    parser.add_argument(
        '--img_id',
        dest='img_id',
        type=int,
        help='Include here desired image id.',
        default=-1
    )

    parser.add_argument(
        '--img_dir',
        dest='img_dir',
        type=str,
        help='Include here the path to the curresponding image folder.',
        default=''
    )

    parser.add_argument(
        '--first',
        dest='first',
        type=int,
        help='visualize only specified amount of gt-images',
        default=-1
    )

    parser.add_argument(
        '--output_dir',
        dest='output_dir',
        type=str,
        help='output directory',
        default=""
    )

    parser.add_argument(
        '--show_label',
        dest='show_label',
        type=bool,
        help='boolean for showing label texts',
        default=False
    )

    parser.add_argument(
        '--ext',
        dest='ext',
        type=str,
        help='file extension visualization',
        default="png"
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def xywh_to_xyxy(box):
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def category_id_to_name(cats, id):
    if id in cats:
        return cats[id]['name']
    else:
        return 'id_not_found'


def anns_to_boxes(anns):
    if len(anns) == 0:
        boxes_list=[[0,0,0,0,-1]]
    else:
        boxes_xyhwh = [ann['bbox'] for ann in anns]
        classes = [ann['category_id'] for ann in anns]
        boxes_list = [xywh_to_xyxy(box)+[classes[ind]]for ind, box in enumerate(boxes_xyhwh)]
    return np.array(boxes_list)


def visualize_one_gt_image(img, img_name, output_dir, boxes, cats,
                           dpi=200, ext='png', show_cls=False):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(img.shape[1] / dpi, img.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(img)

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    for i in sorted_inds:
        bbox = boxes[i, :4]
        class_id = boxes[i, -1]
        if class_id != -1:
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1],
                              fill=False, edgecolor=cfg.VIS.GT_COLOR,
                              linewidth=cfg.VIS.BOX.LINEWIDTH, alpha=cfg.VIS.BOX.ALPHA))

            # do not plot not matched detections
            # if gt-boxes drawn: show_classes always for wrong (red) detections
            if cfg.VIS.GT_SHOW_CLASS or show_cls:
                ax.text(
                    bbox[0] + 1, bbox[1] - 6,
                    category_id_to_name(cats, class_id),
                    fontsize=cfg.VIS.LABEL.FONTSIZE,
                    family=cfg.VIS.LABEL.FAMILY, weight=cfg.VIS.LABEL.WEIGHT,
                    bbox=dict(
                        facecolor=cfg.VIS.GT_COLOR, alpha=cfg.VIS.LABEL.ALPHA, pad=cfg.VIS.LABEL.PAD, edgecolor='none'),
                    color=cfg.VIS.LABEL.GT_TEXTCOLOR)

    output_name = os.path.basename(img_name) + '.' + ext
    fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
    plt.close('all')


def main():
    args = parse_args()
    if args.json_file=='':
        raise ValueError('No json_file specified.')
    if args.img_id==-1 and args.first==-1:
        raise ValueError('No images to visualize specified')
    if args.img_dir=='':
        raise ValueError('No img_dir specified')
    if args.output_dir=='':
        raise ValueError('No output_dir specified')
    if not os.path.exists(args.json_file):
        raise ValueError('Specified json_file does not exist')

    coco = COCO(args.json_file)

    if args.img_id != -1:
        img_ids = coco.getImgIds(imgIds=args.img_id)
    # else first specified
    else:
        img_ids = coco.getImgIds()
        img_ids = [img_ids[rand_int] for rand_int in np.random.randint(0, len(img_ids), args.first)]

    imgs = coco.loadImgs(ids=img_ids)

    for i, img in enumerate(imgs):
        if i % 10 == 0:
            print("{}/{}".format(i+1, len(imgs)))
        if os.path.exists(args.img_dir + '/' + img['file_name']):
            img_path = args.img_dir + '/' + img['file_name']
        else:
            raise ValueError('Desired image does not exist')
        im = cv2.imread(img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_name = img['file_name'].split('.')[0] + '_gt_boxes'

        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        boxes = anns_to_boxes(anns)

        visualize_one_gt_image(im, im_name, args.output_dir, boxes, coco.cats,
                               ext=args.ext, show_cls=args.show_label)


if __name__ == '__main__':
    main()