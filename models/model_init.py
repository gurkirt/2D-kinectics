
from .inceptionv3 import inception_v3
from .vgg import vggnet
import torch

def initialise_model(args):
    # create model
    if args.arch.find('inceptionV3') > -1:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = inception_v3(num_classes=args.num_classes, pretrained=True, global_models_dir=args.global_models_dir, seq_len=args.seq_len)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = inception_v3(num_classes=args.num_classes, seq_len=args.seq_len)
    elif args.arch.find('vgg') > -1:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = vggnet(num_classes=args.num_classes, pretrained=True, global_models_dir=args.global_models_dir, seq_len=args.seq_len)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = vggnet(num_classes=args.num_classes, seq_len=args.seq_len)
    else:
        raise 'Spcify the correct model type'

    if args.ngpu>1:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
        else:
            print('Apply DataParallel')
            model = torch.nn.DataParallel(model)

    model.cuda()
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    return model, criterion