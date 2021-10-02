from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from yolov3_tf2.new_models import YoloV3, YoloV3Tiny
from yolov3_tf2.utils import load_darknet_weights
from yolov3_tf2.mobilenet_utils import load_mobilenet_weights
import tensorflow as tf

flags.DEFINE_string('weights', './data/yolov3.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov3.tf', 'path to output')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_enum('backbone', 'darknet',
                  ['darknet', 'tiny', 'mobilenet'],
                  'darknet: Transfer darknet for yolov3, '
                  'tiny: Transfer darknet for yolov3-tiny, '
                  'mobilenet: Transfer mobilenet, ')

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(FLAGS.backbone)
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.backbone != 'mobilenet':
        if FLAGS.backbone == 'tiny':
            yolo = YoloV3Tiny(classes=FLAGS.num_classes)
        elif FLAGS.backbone == 'darknet':
            yolo = YoloV3(classes=FLAGS.num_classes)
        
        yolo.summary()
        logging.info('model created')

        load_darknet_weights(yolo, FLAGS.weights, FLAGS.backbone=='tiny')
        logging.info('weights loaded')

    else:
        yolo = load_mobilenet_weights(FLAGS.weights, FLAGS.num_classes)
        yolo.summary()
        logging.info('weights loaded')
        

    img = np.random.random((1, 416, 416, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')

    yolo.save_weights(FLAGS.output)
    logging.info('weights saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
