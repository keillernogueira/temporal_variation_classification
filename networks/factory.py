from networks.segnet import SegNet
from networks.deeplab import DeepLabV3Plus


def model_factory(model_name, num_input_bands, num_classes):
    if model_name == 'segnet':
        return SegNet(num_input_bands, num_classes)
    elif model_name == 'deeplab':
        return DeepLabV3Plus(num_input_bands, num_classes)
    else:
        raise NotImplementedError('Network not identified: ' + model_name)
