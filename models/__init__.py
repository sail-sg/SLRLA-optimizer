
def get_networks(name):
    if name == 'WRN-28-10':
        from .nets.wide_resnet import WRN_28_10
        network = WRN_28_10
        
    elif name == 'WRN-16-10':
        from .nets.wide_resnet import WRN_16_10
        network = WRN_16_10
        
    elif name == 'WRN-16-8':
        from .nets.wide_resnet import WRN_16_8
        network = WRN_16_8

    elif name == 'ResNet-10':
        from .nets.resnet import ResNet10
        network = ResNet10
        
    elif name == 'ResNet-12':
        from .nets.resnet import ResNet12
        network = ResNet12
        
    elif name == 'ResNet-18':
        from .nets.resnet import ResNet18
        network = ResNet18
        
    elif name == 'ResNet-34':
        from .nets.resnet import ResNet34
        network = ResNet34
        
    elif name == 'ResNet-50':
        from .nets.resnet import ResNet50
        network = ResNet50
        
    elif name == 'VGG-11':
        from .nets.vgg import VGG_11
        network = VGG_11
        
    elif name == 'VGG-19':
        from .nets.vgg import VGG_19
        network = VGG_19
        
    elif name == 'VGG-16':
        from .nets.vgg import VGG_16
        network = VGG_16
        
    else:
        assert False
    return network