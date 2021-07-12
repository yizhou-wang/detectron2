# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import pycocotools.mask as cocomask

cruw_data_root = '/mnt/disk1/CRUW/CRUW_MINI/sequences'
cruw_infer_root = '/mnt/disk1/CRUW/CRUW_MINI/maskrcnn_results'

CITYSCAPES_THINGS = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


def visualize(im, instance, save_path=None):
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(instance)
    if save_path is None:
        cv2.imshow("Demo", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    else:
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])


def transform_instance_to_dict(instances):
    scores = instances.scores.numpy()
    pred_classes = instances.pred_classes.numpy()
    pred_boxes = instances.pred_boxes.tensor.numpy()
    pred_masks = instances.pred_masks.numpy()
    pred_masks_encode = []
    for objid in range(len(scores)):
        mask_encode = cocomask.encode(np.asfortranarray(pred_masks[objid]))
        pred_masks_encode.append(mask_encode)
    return {
        'n_obj': len(scores),
        'scores': scores,
        'pred_classes': pred_classes,
        'pred_boxes': pred_boxes,
        'pred_masks': pred_masks_encode
    }


def write_det_txt(save_path, output_dict):
    with open(save_path, 'w') as f:
        for frameid in range(len(output_dict['IMAGES_0'])):
            dets_frame_dict = output_dict['IMAGES_0'][frameid]
            for objid in range(dets_frame_dict['n_obj']):
                mask_encode = dets_frame_dict['pred_masks'][objid]
                det_str = "%d %s %s %.4f %.2f %.2f %.2f %.2f %d %d %s\n" % \
                          (frameid, 'IMAGES_0', CITYSCAPES_THINGS[dets_frame_dict['pred_classes'][objid]],
                           dets_frame_dict['scores'][objid],
                           dets_frame_dict['pred_boxes'][objid][0], dets_frame_dict['pred_boxes'][objid][1],
                           dets_frame_dict['pred_boxes'][objid][2], dets_frame_dict['pred_boxes'][objid][3],
                           mask_encode['size'][0], mask_encode['size'][1], mask_encode['counts'])
                f.write(det_str)

            if len(output_dict['IMAGES_1']) != 0:
                dets_frame_dict = output_dict['IMAGES_1'][frameid]
                for objid in range(dets_frame_dict['n_obj']):
                    mask_encode = dets_frame_dict['pred_masks'][objid]
                    det_str = "%d %s %s %.4f %.2f %.2f %.2f %.2f %d %d %s\n" % \
                              (frameid, 'IMAGES_0', CITYSCAPES_THINGS[dets_frame_dict['pred_classes'][objid]],
                               dets_frame_dict['scores'][objid],
                               dets_frame_dict['pred_boxes'][objid][0], dets_frame_dict['pred_boxes'][objid][1],
                               dets_frame_dict['pred_boxes'][objid][2], dets_frame_dict['pred_boxes'][objid][3],
                               mask_encode['size'][0], mask_encode['size'][1], mask_encode['counts'])
                    f.write(det_str)


if __name__ == '__main__':

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
    predictor = DefaultPredictor(cfg)

    seq_names = sorted(os.listdir(cruw_data_root))
    for seq in seq_names:
        output_dict = {
            'IMAGES_0': [],
            'IMAGES_1': [],
        }
        seq_path = os.path.join(cruw_data_root, seq)

        seq_path0 = os.path.join(seq_path, 'IMAGES_0')
        image_names0 = sorted(os.listdir(seq_path0))
        for image_name in image_names0:
            im_path = os.path.join(seq_path0, image_name)
            print("Inferring %s" % im_path)
            im = cv2.imread(im_path)
            outputs = predictor(im)
            instances = outputs["instances"].to("cpu")
            det_dict = transform_instance_to_dict(instances)
            output_dict['IMAGES_0'].append(det_dict)

            save_path = os.path.join(cruw_infer_root, 'vis', seq, 'IMAGES_0', image_name)
            if not os.path.exists(os.path.join(cruw_infer_root, 'vis', seq, 'IMAGES_0')):
                os.makedirs(os.path.join(cruw_infer_root, 'vis', seq, 'IMAGES_0'))
            visualize(im, instances, save_path)

        seq_path1 = os.path.join(seq_path, 'IMAGES_1')
        if os.path.isdir(seq_path1):
            image_names1 = sorted(os.listdir(seq_path1))
            for image_name in image_names1:
                im_path = os.path.join(seq_path1, image_name)
                print("Inferring %s" % im_path)
                im = cv2.imread(im_path)
                outputs = predictor(im)
                instances = outputs["instances"].to("cpu")
                det_dict = transform_instance_to_dict(instances)
                output_dict['IMAGES_1'].append(det_dict)

                save_path = os.path.join(cruw_infer_root, 'vis', seq, 'IMAGES_1', image_name)
                if not os.path.exists(os.path.join(cruw_infer_root, 'vis', seq, 'IMAGES_1')):
                    os.makedirs(os.path.join(cruw_infer_root, 'vis', seq, 'IMAGES_1'))
                visualize(im, instances, save_path)

        save_path = os.path.join(cruw_infer_root, 'txts', seq + '.txt')
        if not os.path.exists(os.path.join(cruw_infer_root, 'txts')):
            os.makedirs(os.path.join(cruw_infer_root, 'txts'))
        write_det_txt(save_path, output_dict)
