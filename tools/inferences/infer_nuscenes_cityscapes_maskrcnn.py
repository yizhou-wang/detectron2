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

data_root = '/mnt/disk1/nuscenes/samples/CAM_FRONT'
infer_root = '/mnt/disk1/nuscenes/maskrcnn_results/samples/CAM_FRONT'
infer_vis_root = '/mnt/disk1/nuscenes/maskrcnn_results/samples/CAM_FRONT_VIS'

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


def write_det_txt(save_path, det_dict):
    with open(save_path, 'w') as f:
        for objid in range(det_dict['n_obj']):
            mask_encode = det_dict['pred_masks'][objid]
            det_str = "%s %.4f %.2f %.2f %.2f %.2f %d %d %s\n" % \
                      (CITYSCAPES_THINGS[det_dict['pred_classes'][objid]],
                       det_dict['scores'][objid],
                       det_dict['pred_boxes'][objid][0], det_dict['pred_boxes'][objid][1],
                       det_dict['pred_boxes'][objid][2], det_dict['pred_boxes'][objid][3],
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

    img_names = sorted(os.listdir(data_root))
    outputs = []
    for img in img_names:
        im_path = os.path.join(data_root, img)

        print("Inferring %s" % im_path)
        im = cv2.imread(im_path)
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")
        det_dict = transform_instance_to_dict(instances)

        save_path = os.path.join(infer_vis_root, img)
        if not os.path.exists(os.path.join(infer_vis_root)):
            os.makedirs(os.path.join(infer_vis_root))
        visualize(im, instances, save_path)

        save_path = os.path.join(infer_root, img.replace('.jpg', '.txt'))
        if not os.path.exists(os.path.join(infer_root)):
            os.makedirs(os.path.join(infer_root))
        write_det_txt(save_path, det_dict)
