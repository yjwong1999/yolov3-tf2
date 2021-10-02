"""YOLO_v3 Model Defined in Keras."""

import numpy as np
import tensorflow as tf
from typing import List, Tuple
from .other_utils import compose,do_giou_calculate
from .override import mobilenet_v2
from .other_train import AdvLossModel


def MobilenetSeparableConv2D(filters,
                             kernel_size,
                             strides=(1, 1),
                             padding='valid',
                             use_bias=True):
    return compose(
        tf.keras.layers.DepthwiseConv2D(kernel_size,
                                        padding=padding,
                                        use_bias=use_bias,
                                        strides=strides),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(6.),
        tf.keras.layers.Conv2D(filters,
                               1,
                               padding='same',
                               use_bias=use_bias,
                               strides=1), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(6.))


def make_last_layers_mobilenet(x, id, num_filters, out_filters):
    x = compose(
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_' + str(id) + '_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_' + str(id) + '_BN'),
        tf.keras.layers.ReLU(6., name='block_' + str(id) + '_relu6'),
        MobilenetSeparableConv2D(2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same'),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_' + str(id + 1) + '_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_' + str(id + 1) + '_BN'),
        tf.keras.layers.ReLU(6., name='block_' + str(id + 1) + '_relu6'),
        MobilenetSeparableConv2D(2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same'),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_' + str(id + 2) + '_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_' + str(id + 2) + '_BN'),
        tf.keras.layers.ReLU(6., name='block_' + str(id + 2) + '_relu6'))(x)
    y = compose(
        MobilenetSeparableConv2D(2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same'),
        tf.keras.layers.Conv2D(out_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False))(x)
    return x, y


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobilenetConv2D(kernel, alpha, filters):
    last_block_filters = _make_divisible(filters * alpha, 8)
    return compose(
        tf.keras.layers.Conv2D(last_block_filters,
                               kernel,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(6.))


def mobilenetv2_yolo_body(inputs, num_anchors, num_classes, alpha=1.0):
    mobilenetv2 = mobilenet_v2(default_batchnorm_momentum=0.9,
                               alpha=alpha,
                               input_tensor=inputs,
                               include_top=False,
                               weights='imagenet')
    x, y1 = make_last_layers_mobilenet(mobilenetv2.output, 17, 512,
                                       num_anchors * (num_classes + 5))
    x = compose(
        tf.keras.layers.Conv2D(256,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_20_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
        tf.keras.layers.ReLU(6., name='block_20_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    x = tf.keras.layers.Concatenate()([
        x,
        MobilenetConv2D(
            (1, 1), alpha,
            384)(mobilenetv2.get_layer('block_12_project_BN').output)
    ])
    x, y2 = make_last_layers_mobilenet(x, 21, 256,
                                       num_anchors * (num_classes + 5))
    x = compose(
        tf.keras.layers.Conv2D(128,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_24_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9, name='block_24_BN'),
        tf.keras.layers.ReLU(6., name='block_24_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    x = tf.keras.layers.Concatenate()([
        x,
        MobilenetConv2D((1, 1), alpha,
                        128)(mobilenetv2.get_layer('block_5_project_BN').output)
    ])
    x, y3 = make_last_layers_mobilenet(x, 25, 128,
                                       num_anchors * (num_classes + 5))
    y1 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y1')(y1)
    y2 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y2')(y2)
    y3 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y3')(y3)
    return AdvLossModel(mobilenetv2.inputs, [y1, y2, y3])


def yolo_head(feats: tf.Tensor,
              anchors: np.ndarray,
              input_shape: tf.Tensor,
              calc_loss: bool = False
             ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = tf.reshape(tf.constant(anchors), [1, 1, 1, num_anchors, 2])
    grid_shape = tf.shape(feats)[1:3]
    grid_y = tf.tile(tf.reshape(tf.range(0, grid_shape[0]), [-1, 1, 1, 1]),
                     [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(0, grid_shape[1]), [1, -1, 1, 1]),
                     [grid_shape[0], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], -1)
    grid = tf.cast(grid, feats.dtype)

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(
        grid_shape[::-1], feats.dtype)
    box_wh = tf.exp(feats[..., 2:4]) * tf.cast(
        anchors_tensor, feats.dtype) / tf.cast(input_shape[::-1], feats.dtype)
    box_confidence = tf.sigmoid(feats[..., 4:5])
    if calc_loss == True:
        return grid, box_xy, box_wh, box_confidence
    box_class_probs = tf.sigmoid(feats[..., 5:])
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy: tf.Tensor, box_wh: tf.Tensor,
                       input_shape: tf.Tensor, image_shape) -> tf.Tensor:
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = tf.cast(input_shape, box_yx.dtype)
    image_shape = tf.cast(image_shape, box_yx.dtype)
    max_shape = tf.maximum(image_shape[0], image_shape[1])
    ratio = image_shape / max_shape
    boxed_shape = input_shape * ratio
    offset = (input_shape - boxed_shape) / 2.
    scale = image_shape / boxed_shape
    box_yx = (box_yx * input_shape - offset) * scale
    box_hw *= input_shape * scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = tf.concat(
        [
            tf.clip_by_value(box_mins[..., 0:1], 0, image_shape[0]),  # y_min
            tf.clip_by_value(box_mins[..., 1:2], 0, image_shape[1]),  # x_min
            tf.clip_by_value(box_maxes[..., 0:1], 0, image_shape[0]),  # y_max
            tf.clip_by_value(box_maxes[..., 1:2], 0, image_shape[1])  # x_max
        ],
        -1)
    return boxes


def yolo_boxes_and_scores(feats: tf.Tensor, anchors: List[Tuple[float, float]],
                          num_classes: int, input_shape: Tuple[int, int],
                          image_shape) -> Tuple[tf.Tensor, tf.Tensor]:
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(
        feats, anchors, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = tf.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores
