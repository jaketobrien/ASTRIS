# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict, OrderedDict
import logging
import os
import os.path
import re  # Import regular expressions

import cv2
import json_tricks as json
import numpy as np
from torch.utils.data import Dataset

from pycocotools.cocoeval import COCOeval
from utils import zipreader

logger = logging.getLogger(__name__)


class CocoDataset(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where dataset is located.
        dataset (string): Dataset name (train2017, val2017, test2017).
        data_format (string): Data format for reading ('jpg', 'zip').
        transform (callable, optional): A function/transform that takes in an opencv image
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(self, root, dataset, data_format, transform=None,
                 target_transform=None):
        from pycocotools.coco import COCO
        self.name = 'COCO'
        self.root = root
        self.dataset = dataset
        self.data_format = data_format
        self.coco = COCO(self._get_anno_file_name())
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

    def _get_anno_file_name(self):
        # example: root/annotations/person_keypoints_train2017.json
        # image_info_test-dev2017.json
        if 'test' in self.dataset:
            return os.path.join(
                self.root,
                'annotations',
                'image_info_{}.json'.format(self.dataset)
            )
        else:
            return os.path.join(
                self.root,
                'annotations',
                'person_keypoints_{}.json'.format(self.dataset)
            )

    def _get_image_path(self, file_name):
        images_dir = os.path.join(self.root, 'images')
        dataset = 'test2017' if 'test' in self.dataset else self.dataset
        if self.data_format == 'zip':
            return os.path.join(images_dir, dataset) + '.zip@' + file_name
        else:
            return os.path.join(images_dir, dataset, file_name)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). Target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        file_name = coco.loadImgs(img_id)[0]['file_name']

        if self.data_format == 'zip':
            img = zipreader.imread(
                self._get_image_path(file_name),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            img = cv2.imread(
                self._get_image_path(file_name),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def processKeypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            p = keypoints[keypoints[:, 2] > 0][:, :2].mean(axis=0)
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [
                    float(keypoints[i][0]),
                    float(keypoints[i][1]),
                    float(keypoints[i][2])
                ]
        return tmp

    def evaluate(self, cfg, preds, scores, output_dir, *args, **kwargs):
        '''
        Perform evaluation on COCO keypoint task.
        :param cfg: cfg dictionary.
        :param preds: predictions.
        :param scores: scores corresponding to the predictions.
        :param output_dir: output directory.
        :return: an ordered dictionary of evaluation metrics and the average precision.
        '''
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(res_folder, 'keypoints_%s_results.json' % self.dataset)

        kpts = defaultdict(list)
        for idx, _kpts in enumerate(preds):
            # Use the COCO image id directly.
            coco_img_id = self.ids[idx]
            # Also extract numeric portion from the file name (for debugging purposes)
            file_name = self.coco.loadImgs(coco_img_id)[0]['file_name']
            file_id_matches = re.findall(r'\d+', file_name)
            if not file_id_matches:
                raise ValueError("No numeric part found in file_name: {}".format(file_name))
            extracted_file_id = int(file_id_matches[0])
            # Print both values for debugging.
            print("COCO image id: {}   extracted file id: {}".format(coco_img_id, extracted_file_id))
            # Use the COCO image id as the file id.
            file_id = coco_img_id

            for idx_kpt, kpt in enumerate(_kpts):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                kpt = self.processKeypoints(kpt)
                if cfg.DATASET.WITH_CENTER and not cfg.TEST.IGNORE_CENTER:
                    kpt = kpt[:-1]
                kpts[file_id].append({
                    'keypoints': kpt[:, 0:3],
                    'score': scores[idx][idx_kpt],
                    'tags': kpt[:, 3],
                    'image': file_id,
                    'area': area
                })

        # Rescoring and oks nms (here we simply keep all detections)
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            keep = []  # no nms filtering applied in this implementation
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(oks_nmsed_kpts, res_file)
        if 'test' not in self.dataset:
            info_str = self._do_python_keypoint_eval(res_file, res_folder)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if cls != '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []
        num_joints = 17

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints'] for k in range(len(img_kpts))])
            key_points = np.zeros((_key_points.shape[0], num_joints * 3), dtype=float)

            for ipt in range(num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]

            for k in range(len(img_kpts)):
                kpt = key_points[k].reshape((num_joints, 3))
                left_top = np.amin(kpt, axis=0)
                right_bottom = np.amax(kpt, axis=0)
                w = right_bottom[0] - left_top[0]
                h = right_bottom[1] - left_top[1]
                cat_results.append({
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'bbox': list([left_top[0], left_top[1], w, h])
                })
        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)',
                       'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))
        return info_str

