from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
import cv2
import timeit

from yolov3_tf2.new_models import (
    YoloV3, yolo_anchors, yolo_anchor_masks,
)
from yolov3_tf2.utils import freeze_all

flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')               
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('weights_num_classes', 80, 'number of classes in the model')


def annotate(model_pretrained, img_paths, json_path, name):
    # annotate dataset if haven annotate
    if not os.path.isfile(json_path):
        # start annotating
        print('Start annotating...')
        t1 = timeit.default_timer()

        # a dict to store all paths and all annotation
        annot_json  = {
            'name': None,
            'data': None,
        }

        # a list to store all annotation
        all_annot = []

        # iterate all images
        for img_path in img_paths:
            # load image
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (FLAGS.size, FLAGS.size)) / 255.0

            # get the prediction
            boxes, scores, classes, nums = model_pretrained.predict(np.expand_dims(img, 0))
            boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]

            # select detected person (with confident)
            scores_thresh = scores > 0.5
            person_class = classes == 0
            mask = np.logical_and(scores_thresh, person_class)

            # select relevant boxes
            selected_boxes = boxes[mask]

            # the template for box cornet: x1, y1, x2, y2
            box_corner = {
                'x1':0,
                'y1':0,
                'x2':0,
                'y2':0,
            }

            # a list to store all bbox for an image
            all_box_in_an_image = []

            # a unit of annot
            annot_unit = {
                'img_path': None,
                'bboxs': None,
            }

            # start recording
            if (len(selected_boxes)) == 0:
                continue

            for box in selected_boxes:
                # record the box corner coord values
                x1, y1, x2, y2 = box
                box_corner['x1'] = str(x1)
                box_corner['y1'] = str(y1)
                box_corner['x2'] = str(x2)
                box_corner['y2'] = str(y2)
                # record this box corner
                all_box_in_an_image.append(box_corner.copy())

            # record all bbox for this image
            annot_unit['img_path'] = str(img_path)
            annot_unit['bboxs'] = all_box_in_an_image.copy()
            all_annot.append(annot_unit)

        # save all information in the annoation dictionary
        annot_json['name'] = name
        annot_json['data'] = all_annot

        with open(json_path, "w") as out_file:
            json.dump(annot_json, out_file, indent = 6) 

        t2 = timeit.default_timer()
        print('Total time to annotate the dataset: {:.2f} min'.format((t2-t1) / 60))
        print('JSON File "{}" is ready\n'.format(json_path))
    else:
        print('JSON File "{}" is already generated\n'.format(json_path))
        
        
def main(_argv):
    # setup the pretrained model
    model_pretrained = YoloV3(
        FLAGS.size, training=False, classes=FLAGS.weights_num_classes,
        anchors = yolo_anchors, masks = yolo_anchor_masks) 
    model_pretrained.load_weights(FLAGS.weights)
    
    # get the original annotation path
    annot_train_path = './crisis_vision_benchmarks/tasks/disaster_types/consolidated/consolidated_disaster_types_train_final.tsv'
    annot_dev_path = './crisis_vision_benchmarks/tasks/disaster_types/consolidated/consolidated_disaster_types_dev_final.tsv'
    annot_test_path = './crisis_vision_benchmarks/tasks/disaster_types/consolidated/consolidated_disaster_types_test_final.tsv'

    # training data
    df = pd.read_csv(annot_train_path, sep='\t')
    train_img_paths = df['image_path'].tolist()
    
    # dev data
    df = pd.read_csv(annot_dev_path, sep='\t')
    dev_img_paths = df['image_path'].tolist()
    
    # test data
    df = pd.read_csv(annot_test_path, sep='\t')
    test_img_paths = df['image_path'].tolist() 

    # annotate    
    img_paths_collection = [train_img_paths, 
                            dev_img_paths, 
                            test_img_paths]        
    json_filenames = ['./others/train_damage_severity_person.json', 
                      './others/val_damage_severity_person.json', 
                      './others/test_damage_severity_person.json']
    names = ['training',
             'validation',
             'testing']
    
    if not os.path.isdir('./others'):
        os.mkdir('./others')
    
    for img_paths, json_filename, name in zip(img_paths_collection, json_filenames, names):
        annotate(model_pretrained, img_paths, json_filename, name)    

        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
