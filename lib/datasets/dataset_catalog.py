# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from core.config import cfg

# Path to data dir
_DATA_DIR = cfg.DATA_DIR

# Required dataset entry keys
_IM_DIR = 'image_directory'
_ANN_FN = 'annotation_file'

# Optional dataset entry keys
_IM_PREFIX = 'image_prefix'
_DEVKIT_DIR = 'devkit_directory'
_RAW_DIR = 'raw_dir'

# Available datasets
# Available datasets
_DATASETS = {
    'cityscapes_fine_instanceonly_seg_train': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_train.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_val': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        # use filtered validation as there is an issue converting contours
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_test': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_test.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'coco_2014_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2014.json'
    },
    'coco_2014_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2014.json'
    },
    'coco_2014_minival': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_minival2014.json'
    },
    'coco_2014_valminusminival': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_valminusminival2014.json'
    },
    'coco_2015_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'coco_2015_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'coco_2017_test': {  # 2017 test uses 2015 test images
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        _IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_2017_test-dev': {  # 2017 test-dev uses 2015 test images
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
        _IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_stuff_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/coco_stuff_train.json'
    },
    'coco_stuff_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/coco_stuff_val.json'
    },
    'keypoints_coco_2014_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2014.json'
    },
    'keypoints_coco_2014_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2014.json'
    },
    'keypoints_coco_2014_minival': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_minival2014.json'
    },
    'keypoints_coco_2014_valminusminival': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_valminusminival2014.json'
    },
    'keypoints_coco_2015_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'keypoints_coco_2015_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'voc_2007_trainval': {
        _IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_trainval.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2007_test': {
        _IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_test.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2012_trainval': {
        _IM_DIR:
            _DATA_DIR + '/VOC2012/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2012/annotations/voc_2012_trainval.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2012/VOCdevkit2012'
    },
    'tt100k_trainval':  {
        _IM_DIR:
            _DATA_DIR + '/tt100k/train',
        _ANN_FN:
            _DATA_DIR + '/tt100k/annotations/tt100k_train.json'
    },
    'tt100k_trainval_ignore_complete': {
        _IM_DIR:
            _DATA_DIR + '/tt100k/train',
        _ANN_FN:
            _DATA_DIR + '/tt100k/annotations/tt100k_train_ignore_complete.json'
    },
    'tt100k_trainval_ignore': {
        _IM_DIR:
            _DATA_DIR + '/tt100k/train',
        _ANN_FN:
            _DATA_DIR + '/tt100k/annotations/tt100k_train_ignore.json'
    },
    'tt100k_test': {
        _IM_DIR:
            _DATA_DIR + '/tt100k/test',
        _ANN_FN:
            _DATA_DIR + '/tt100k/annotations/tt100k_test.json'
    },
    'tt100k_test_ignore': {
        _IM_DIR:
            _DATA_DIR + '/tt100k/test',
        _ANN_FN:
            _DATA_DIR + '/tt100k/annotations/tt100k_test_ignore.json'
    },
    'tt100k_test_ignore_complete': {
        _IM_DIR:
            _DATA_DIR + '/tt100k/test',
        _ANN_FN:
            _DATA_DIR + '/tt100k/annotations/tt100k_test_ignore_complete.json'
    },
    'tt100k_val_small_ignore': {
        _IM_DIR:
            _DATA_DIR + '/tt100k/other',
        _ANN_FN:
            _DATA_DIR + '/tt100k/annotations/tt100k_other_ignore.json'
    },
    'tt100k_val_small_ignore_complete': {
        _IM_DIR:
            _DATA_DIR + '/tt100k/other',
        _ANN_FN:
            _DATA_DIR + '/tt100k/annotations/tt100k_other_ignore_complete.json'
    },
    'kitti_trainval': {
        _IM_DIR:
            _DATA_DIR + '/kitti/training/image_2',
        _ANN_FN:
            _DATA_DIR + '/tt100k/annotations/kitti_train.json'
    },
    'kitti_test': {
        _IM_DIR:
            _DATA_DIR + '/kitti/training/image_2',
        _ANN_FN:
            _DATA_DIR + '/kitti/annotations/kitti_test.json'
    },
    'kitti_trainval_ignore': {
        _IM_DIR:
            _DATA_DIR + '/kitti/training/image_2',
        _ANN_FN:
            _DATA_DIR + '/tt100k/annotations/kitti_train_ignore.json'
    },
    'kitti_test_ignore': {
        _IM_DIR:
            _DATA_DIR + '/kitti/training/image_2',
        _ANN_FN:
            _DATA_DIR + '/kitti/annotations/kitti_test_ignore.json'
    },
    'kitti_trainval_ignore_complete': {
        _IM_DIR:
            _DATA_DIR + '/kitti/training/image_2',
        _ANN_FN:
            _DATA_DIR + '/tt100k/annotations/kitti_train_ignore_complete.json'
    },
    'kitti_test_ignore_complete': {
        _IM_DIR:
            _DATA_DIR + '/kitti/training/image_2',
        _ANN_FN:
            _DATA_DIR + '/kitti/annotations/kitti_test_ignore_complete.json'
    },
    'vkitti_new_clone_trainval': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_clone_train.json'
    },
    'vkitti_new_clone_test': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_clone_test.json'
    },
    'vkitti_new_clone_all': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_clone_all.json'
    },
    'vkitti_new_rain_trainval': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_rain_train.json'
    },
    'vkitti_new_rain_test': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_rain_test.json'
    },
    'vkitti_new_rain_all': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_rain_all.json'
    },
    'vkitti_new_morning_trainval': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_morning_train.json'
    },
    'vkitti_new_morning_test': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_morning_test.json'
    },
    'vkitti_new_morning_all': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_morning_all.json'
    },
    'vkitti_new_fog_trainval': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_fog_train.json'
    },
    'vkitti_new_fog_test': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_fog_test.json'
    },
    'vkitti_new_fog_all': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_fog_all.json'
    },
    'vkitti_new_overcast_trainval': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_overcast_train.json'
    },
    'vkitti_new_overcast_test': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_overcast_test.json'
    },
    'vkitti_new_overcast_all': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_overcast_all.json'
    },
    'vkitti_new_sunset_trainval': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_sunset_train.json'
    },
    'vkitti_new_sunset_test': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_sunset_test.json'
    },
    'vkitti_new_sunset_all': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_new_sunset_all.json'
    },
    'vkitti_clone_trainval': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_clone_train.json'
    },
    'vkitti_clone_test': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_clone_test.json'
    },
    'vkitti_clone_all': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_clone_all.json'
    },
    'vkitti_rain_trainval': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_rain_train.json'
    },
    'vkitti_rain_test': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_rain_test.json'
    },
    'vkitti_rain_all': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_rain_all.json'
    },
    'vkitti_morning_trainval': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_morning_train.json'
    },
    'vkitti_morning_test': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_morning_test.json'
    },
    'vkitti_morning_all': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_morning_all.json'
    },
    'vkitti_fog_trainval': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_fog_train.json'
    },
    'vkitti_fog_test': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_fog_test.json'
    },
    'vkitti_fog_all': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_fog_all.json'
    },
    'vkitti_overcast_trainval': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_overcast_train.json'
    },
    'vkitti_overcast_test': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_overcast_test.json'
    },
    'vkitti_overcast_all': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_overcast_all.json'
    },
    'vkitti_sunset_trainval': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_sunset_train.json'
    },
    'vkitti_sunset_test': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_sunset_test.json'
    },
    'vkitti_sunset_all': {
        _IM_DIR:
            _DATA_DIR + '/vkitti/Images',
        _ANN_FN:
            _DATA_DIR + '/vkitti/annotations/vkitti_sunset_all.json'
    },
    'caltech_original_trainval': {
        _IM_DIR:
            _DATA_DIR + '/caltech_pedestrian/train',
        _ANN_FN:
            _DATA_DIR + '/caltech_pedestrian/annotations/caltech_original_train.json'
    },
    'caltech_original_test': {
        _IM_DIR:
            _DATA_DIR + '/caltech_pedestrian/test',
        _ANN_FN:
            _DATA_DIR + '/caltech_pedestrian/annotations/caltech_original_test.json'
    },
    'caltech_dense_trainval': {
        _IM_DIR:
            _DATA_DIR + '/caltech_pedestrian/train',
        _ANN_FN:
            _DATA_DIR + '/caltech_pedestrian/annotations/caltech_dense_train.json'
    },
    'caltech_dense_test': {
        _IM_DIR:
            _DATA_DIR + '/caltech_pedestrian/test',
        _ANN_FN:
            _DATA_DIR + '/caltech_pedestrian/annotations/caltech_dense_test.json'
    },
    'caltech_new_trainval': {
        _IM_DIR:
            _DATA_DIR + '/caltech_pedestrian/train',
        _ANN_FN:
            _DATA_DIR + '/caltech_pedestrian/annotations/caltech_new_train.json'
    },
    'caltech_new_test': {
        _IM_DIR:
            _DATA_DIR + '/caltech_pedestrian/test',
        _ANN_FN:
            _DATA_DIR + '/caltech_pedestrian/annotations/caltech_new_test.json'
    }
}