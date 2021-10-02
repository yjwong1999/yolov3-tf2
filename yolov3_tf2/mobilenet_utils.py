import tensorflow as tf
import os
import numpy as np

try:
    import neural_structured_learning as nsl
except:
    os.system('pip install --upgrade neural_structured_learning')

from .new_models import MobilenetYoloV3
from .other_model import mobilenetv2_yolo_body
from .other_utils import ModelFactory

# names of layers in the reference model to be copied from
CONV_0_REF_LAYERS = [
                      'block_17_conv','block_17_BN','block_17_relu6',
                      'depthwise_conv2d','batch_normalization','re_lu',
                      'conv2d','batch_normalization_1','re_lu_1',
                      'block_18_conv','block_18_BN','block_18_relu6',
                      'depthwise_conv2d_1','batch_normalization_2','re_lu_2',
                      'conv2d_1','batch_normalization_3','re_lu_3',
                      'block_19_conv','block_19_BN','block_19_relu6'
                      ]
CONV_1_REF_LAYERS = [
                      'block_20_conv','block_20_BN','block_20_relu6',
                      'up_sampling2d','concatenate','block_21_conv',
                      'block_21_BN','block_21_relu6','depthwise_conv2d_3',
                      'batch_normalization_7','re_lu_7','conv2d_5',
                      'batch_normalization_8','re_lu_8','block_22_conv',
                      'block_22_BN','block_22_relu6','depthwise_conv2d_4',
                      'batch_normalization_9','re_lu_9','conv2d_6',
                      'batch_normalization_10','re_lu_10','block_23_conv',
                      'block_23_BN','block_23_relu6'
                      ]
CONV_2_REF_LAYERS = [
                      'block_24_conv','block_24_BN','block_24_relu6',
                      'up_sampling2d_1','concatenate_1','block_25_conv',
                      'block_25_BN','block_25_relu6','depthwise_conv2d_6',
                      'batch_normalization_14','re_lu_14','conv2d_10',
                      'batch_normalization_15','re_lu_15','block_26_conv',
                      'block_26_BN','block_26_relu6','depthwise_conv2d_7',
                      'batch_normalization_16','re_lu_16','conv2d_11',
                      'batch_normalization_17','re_lu_17','block_27_conv',
                      'block_27_BN','block_27_relu6',
]
OUT_0_REF_LAYERS = [
                    'depthwise_conv2d_2','batch_normalization_4','re_lu_4',
                    'conv2d_2', 'batch_normalization_5', 're_lu_5', 
                    'conv2d_3','y1'
]
OUT_1_REF_LAYERS = [
                    'depthwise_conv2d_5','batch_normalization_11','re_lu_11',
                    'conv2d_7', 'batch_normalization_12', 're_lu_12', 
                    'conv2d_8','y2'
]
OUT_2_REF_LAYERS = [
                    'depthwise_conv2d_8','batch_normalization_18','re_lu_18',
                    'conv2d_12', 'batch_normalization_19', 're_lu_19', 
                    'conv2d_13', 'y3'
]


# paths of converted path
backbone_path = '../checkpoints/pretrained_mobilenet'


# get the pretrained backbone weights
def __convert_backbone(ref_model):
    # get original mobilenet
    base_model = ref_model

    # Use the activations of these layers
    layer_names = [
        're_lu_13',   # 16x16
        're_lu_6',  # 8x8
        'out_relu',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    backbone = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs, name='mobilenet')
    backbone.save_weights(backbone_path)


# get the pretrained mobilenet-yolo weights
def load_mobilenet_weights(ref_model_path, num_classes):
    # Load the trained Mobilenetv2-YOLO (diff implementation)
    input_shape = (416, 416)
    factory = ModelFactory(tf.keras.layers.Input(shape=(*input_shape, 3)),
                            weights_path=ref_model_path)

    ref_model = factory.build(mobilenetv2_yolo_body,
                          155,
                          3,
                          20,
                          alpha=1.4)    

    # this is the modularize model (the mobilenet-yolo implementation in this repo)
    model = MobilenetYoloV3(classes=num_classes, training=True)

    # get the weights for the backbone
    __convert_backbone(ref_model)
    model.get_layer('mobilenet').load_weights(backbone_path)

    # dictionaries to keep several [layer names in each module]
    ALL_YOLO_MODULES_LAYERS = {
        'input': None,
        'mobilenet': None,
        'yolo_conv_0': None,
        'yolo_conv_1': None,
        'yolo_conv_2': None,
        'yolo_output_0': None,
        'yolo_output_1': None,
        'yolo_output_2': None,
    }
    ALL_SUB_MODULES_MASK = {
        'input': None,
        'mobilenet': None,    
        'yolo_conv_0': None,
        'yolo_conv_1': None,
        'yolo_conv_2': None,
        'yolo_output_0': None,
        'yolo_output_1': None,
        'yolo_output_2': None,
    }
    ALL_REFERENCES_LAYERS = {
        'input': None,
        'mobilenet': None,
        'yolo_conv_0': None,
        'yolo_conv_1': None,
        'yolo_conv_2': None,
        'yolo_output_0': None,
        'yolo_output_1': None,
        'yolo_output_2': None,
    }

    # get [ [layer names in each module], [...] ] except input and backbone
    YOLO_MODULES = [layer.name for layer in model.layers]
    for module_name in YOLO_MODULES:
        YOLO_MODULES_LAYERS = []
        SUB_MODULES_MASK = []
        module = model.get_layer(module_name)
        if type(module) == type(model):
            for layer in module.layers:
                if layer.name.startswith('model'):
                    sub_module = layer
                    SUB_MODULES_MASK.append([sub_module.name, len(sub_module.layers)-1])
                    for layer in sub_module.layers:
                        if not layer.name.startswith('input'):
                            YOLO_MODULES_LAYERS.append(layer.name)        
                elif not layer.name.startswith('input'):
                    YOLO_MODULES_LAYERS.append(layer.name)  
                    SUB_MODULES_MASK.append(None)
            
            ALL_YOLO_MODULES_LAYERS[module_name] = YOLO_MODULES_LAYERS
            ALL_SUB_MODULES_MASK[module_name] = SUB_MODULES_MASK

    ALL_REFERENCES_LAYERS['yolo_conv_0'] = CONV_0_REF_LAYERS
    ALL_REFERENCES_LAYERS['yolo_conv_1'] = CONV_1_REF_LAYERS
    ALL_REFERENCES_LAYERS['yolo_conv_2'] = CONV_2_REF_LAYERS
    ALL_REFERENCES_LAYERS['yolo_output_0'] = OUT_0_REF_LAYERS
    ALL_REFERENCES_LAYERS['yolo_output_1'] = OUT_1_REF_LAYERS
    ALL_REFERENCES_LAYERS['yolo_output_2'] = OUT_2_REF_LAYERS

    # iterate all modules in yolov3 except input and backbone
    for MODULE_NAME in YOLO_MODULES[:-3]:
        # get the list of (1) layers names in the modules
        YOLO_MODULES_LAYERS = ALL_YOLO_MODULES_LAYERS[MODULE_NAME]
        # get the list of (2) mask for submodules
        SUB_MODULES_MASK = ALL_SUB_MODULES_MASK[MODULE_NAME]
        # get the list of (3) references layers name in the module
        REFERENCES_LAYERS = ALL_REFERENCES_LAYERS[MODULE_NAME]
        # continue next iteration if None is acquired
        if YOLO_MODULES_LAYERS is None or SUB_MODULES_MASK is None or REFERENCES_LAYERS is None:
            print(f'No transfer learning for {MODULE_NAME}')
            print('---------------------------------------------------\n')
            continue
        # make the mask an iterable
        SUB_MODULES_MASK = iter(SUB_MODULES_MASK)
        mask = next(SUB_MODULES_MASK)
        # test
        print(f'Transfer learning for {MODULE_NAME}')
        print('---------------------------------------------------')

        assert len(REFERENCES_LAYERS) == len(YOLO_MODULES_LAYERS), f"Reference layers for '{MODULE_NAME}' are not compatible"

        for ref_layer, layer in zip(REFERENCES_LAYERS, YOLO_MODULES_LAYERS):
            print('{:25} -> {:25}'.format(ref_layer, layer))
            if mask is None:
                model.get_layer(MODULE_NAME).get_layer(layer).set_weights(
                    ref_model.get_layer(ref_layer).get_weights())
                try:
                    mask = next(SUB_MODULES_MASK)
                except:
                    break
            else:
                sub_module, _ = mask
                model.get_layer(MODULE_NAME).get_layer(sub_module).get_layer(layer).set_weights(
                    ref_model.get_layer(ref_layer).get_weights())
                mask[1] -= 1
                if mask[1] == 0:
                    try:
                        mask = next(SUB_MODULES_MASK)
                    except:
                        break
        print()

    # save the final weights
    # model.save_weights(mobilenet_yolo_path) # will be saved in convert.py

    return model
