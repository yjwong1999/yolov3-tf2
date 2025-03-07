from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
import cv2
import time
import os
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.new_models import (
    MobilenetYoloV3, YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
from multitask.utils import get_annotation
from multitask import victim_dataset_utils as dataset

# flags.DEFINE_string('dataset', '', 'path to dataset')
# flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('backbone', 'darknet',
                  ['darknet', 'tiny', 'mobilenet'],
                  'darknet: darknet53, '
                  'tiny: tiny darknet, '
                  'mobilenet: Transfer all and freeze darknet only')
flags.DEFINE_enum('transfer', 'no_output',
                  ['none', 'no_output'],
                  'none: train from scratch, '
                  'no_output: Transfer and freeze all layers except output')                  
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')
flags.DEFINE_boolean('multi_gpu', False, 'Use if wishing to train with more than 1 GPU.')
flags.DEFINE_string('checkpoints', '', 'the directory to save checkpoints during training')



def setup_model():
    if FLAGS.backbone == 'tiny':
        model = YoloV3Tiny(FLAGS.size, training=True,
                          classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        if FLAGS.backbone == 'darknet':
            model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        elif FLAGS.backbone == 'mobilenet':
            model = MobilenetYoloV3(
                      FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks
    
    if FLAGS.transfer == 'none':
        # do nothing
        pass 
    elif FLAGS.transfer == 'no_output':
        # get pretrained network
        # only darknet works for different input size
        if FLAGS.backbone == 'tiny':
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        elif FLAGS.backbone == 'darknet':
            model_pretrained = YoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        else:
            model_pretrained = MobilenetYoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)                
        model_pretrained.load_weights(FLAGS.weights)
        # transfer weights and freeze them
        for l in model.layers:
            if not l.name.startswith('yolo_output'):
                l.set_weights(model_pretrained.get_layer(
                    l.name).get_weights())
                freeze_all(l)      

    # get optimizer, loss
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss,
                  run_eagerly=(FLAGS.mode == 'eager_fit'))

    return model, optimizer, loss, anchors, anchor_masks


def get_dataset():
    # hard code some parameters
    MAX_LIMIT = None
    train_json_path = './others/train_damage_severity_person.json'
    val_json_path = './others/val_damage_severity_person.json'
    test_json_path = './others/test_damage_severity_person.json'
    for path in [train_json_path, val_json_path, test_json_path]:
        warning_message = f"'{path} does not exist! Please run ./setup.ipynb !"
        assert os.path.isfile(path), warning_message
    
    # get the img paths and annotations
    train_img_paths, train_annots = get_annotation(train_json_path, MAX_LIMIT)
    val_img_paths, val_annots = get_annotation(val_json_path, MAX_LIMIT)
    test_img_paths, test_annots = get_annotation(test_json_path, MAX_LIMIT)
    
    # report the size of training/test/split
    logging.info(f'Total training data: {len(train_annots)}')
    logging.info(f'Total training data: {len(val_annots)}')
    logging.info(f'Total training data: {len(test_annots)}')
    
    # convert the paths/annotation into tensor
    train_img_paths = tf.convert_to_tensor(train_img_paths, dtype=tf.string)
    train_annots = tf.convert_to_tensor(train_annots, dtype=tf.float32) 
    
    val_img_paths = tf.convert_to_tensor(val_img_paths, dtype=tf.string)
    val_annots = tf.convert_to_tensor(val_annots, dtype=tf.float32) 
    
    test_img_paths = tf.convert_to_tensor(test_img_paths, dtype=tf.string)
    test_annots = tf.convert_to_tensor(test_annots, dtype=tf.float32) 
    
    # convert to tf.data
    trainDS = tf.data.Dataset.from_tensor_slices((train_img_paths, train_annots))
    valDS = tf.data.Dataset.from_tensor_slices((val_img_paths, val_annots))
    testDS = tf.data.Dataset.from_tensor_slices((test_img_paths, test_annots))
    
    return trainDS, valDS, testDS


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    # Setup
    if FLAGS.multi_gpu:
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        BATCH_SIZE = FLAGS.batch_size * strategy.num_replicas_in_sync
        FLAGS.batch_size = BATCH_SIZE

        with strategy.scope():
            model, optimizer, loss, anchors, anchor_masks = setup_model()
    else:
        model, optimizer, loss, anchors, anchor_masks = setup_model()

    # get dataset
    assert FLAGS.checkpoints != '', 'Must specified a checkpoint path'
    train_dataset, val_dataset, test_dataset = get_dataset()

    # transform dataset
    # (1) training dataset
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.load_image(x, FLAGS.size, augmentation=False),
        dataset.transform_target(y, anchors, anchor_masks, FLAGS.size)))    
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.AUTOTUNE)
    
    # (2) validation dataset
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.load_image(x, FLAGS.size, augmentation=False),
        dataset.transform_target(y, anchors, anchor_masks, FLAGS.size)))
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    
    # training
    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                logging.info("{}_val_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_val_loss.update_state(total_loss)

            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights(
                '{}/yolov3_train_{}.tf'.format(FLAGS.checkpoints, epoch))
    else:
        checkpoint = '{}/yolov3_train'.format(FLAGS.checkpoints) + '_{epoch}.tf'
        callbacks = [
            ReduceLROnPlateau(patience=3, verbose=1),
            EarlyStopping(patience=4, verbose=1),
            ModelCheckpoint(checkpoint,
                            verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]

        start_time = time.time()
        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)
        end_time = time.time() - start_time
        print(f'Total Training Time: {end_time}')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
