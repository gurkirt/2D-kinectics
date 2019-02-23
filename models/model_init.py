
from .inceptionv3 import inception_v3
from .vgg import vggnet
from .resnet import resnet
import torch

def initialise_model(args):
    # create model
    if args.arch.find('inceptionV3') > -1:
        if args.pretrained>0:
            print("=> using pre-trained model '{}'".format(args.arch))
            print("NUmber of classes will be ", args.num_classes)
            model = inception_v3(num_classes=args.num_classes, pretrained=True, global_models_dir=args.global_models_dir, seq_len=args.seq_len)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = inception_v3(num_classes=args.num_classes, seq_len=args.seq_len)
    elif args.arch.find('vgg') > -1:
        if args.pretrained>0:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = vggnet(num_classes=args.num_classes, pretrained=True, global_models_dir=args.global_models_dir, seq_len=args.seq_len)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = vggnet(num_classes=args.num_classes, seq_len=args.seq_len)
    elif args.arch[:6] == 'resnet':
        modelperms = {'resnet18': [2, 2, 2, 2], 'resent34': [3, 4, 6, 3], 'resnet50': [3, 4, 6, 3],
                      'resnet101': [3, 4, 23, 3], 'resent152': [3, 8, 36, 3]}
        model = resnet(modelperms[args.arch], args.arch, args.seq_len, args.num_classes)
        print('pretrained:: ', args.pretrained)
        if args.pretrained>0:
            load_dict = torch.load(args.global_models_dir + '/' + args.arch+'.pth')
            # print(load_dict.keys(), '\n\n', model.state_dict().keys())
            model.load_my_state_dict(load_dict, args.seq_len)
    else:
        raise Exception('Spcify the correct model type')

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