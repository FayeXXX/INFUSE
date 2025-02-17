import os
import argparse
import datetime
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from optim import build_optimizer, build_lr_scheduler
from methods.ARPL.core.train_cocoop import train
from methods.ARPL.core.test_cocoop import test
from utils.cocoop_utils import init_experiment, seed_torch, str2bool, get_default_hyperparameters
from data.open_set_datasets import get_class_splits, get_datasets
from utils.logger import Log
from config import exp_root
import sys
import errno
import os.path as osp
import torch.nn as nn
parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='cub', help="")
parser.add_argument('--out_num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--image_size', type=int, default=224)

# optimization
# parser.add_argument('--optim', type=str, default=None, help="Which optimizer to use {adam, sgd}")
# parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
# parser.add_argument('--weight_decay', type=float, default=1e-4, help="LR regularisation on weights")
# parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
# parser.add_argument('--max-epoch', type=int, default=100)
# parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts')
# parser.add_argument('--temp', type=float, default=1.0, help="temp")
# parser.add_argument('--num_restarts', type=int, default=2, help='How many restarts for cosine_warm_restarts schedule')
# parser.add_argument('--num-centers', type=int, default=1)
# cocoop optimization
parser.add_argument('--MAX_EPOCH', type=int, default=5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--NAME', type=str, default='sgd')
parser.add_argument('--LR', type=float,  default=0.002)
parser.add_argument('--LR_SCHEDULER', type=str, default='cosine')
parser.add_argument('--WARMUP_EPOCH', type=int, default=1)
parser.add_argument('--WARMUP_TYPE', type=str, default='constant')
parser.add_argument('--WARMUP_CONS_LR', type=float,  default=1e-5)
parser.add_argument('--WEIGHT_DECAY', type=float,  default=5e-4)
parser.add_argument('--MOMENTUM', type=float,  default=0.9)
parser.add_argument('--SGD_DAMPNING', type=int, default=0)
parser.add_argument('--SGD_NESTEROV', type=str2bool, default=False)
parser.add_argument('--RMSPROP_ALPHA', type=float,  default=0.99)
parser.add_argument('--ADAM_BETA1', type=float,  default=0.9)
parser.add_argument('--ADAM_BETA2', type=float,  default=0.999)
parser.add_argument('--STAGED_LR', type=str2bool, default=False)
parser.add_argument('--NEW_LAYERS', type=tuple, default=())
parser.add_argument('--BASE_LR_MULT', type=float,  default=0.1)
parser.add_argument('--STEPSIZE', type=tuple,  default=(-1, ))
parser.add_argument('--GAMMA', type=float,  default=0.1)
parser.add_argument('--WARMUP_RECOUNT', type=str2bool, default=True)
parser.add_argument('--WARMUP_MIN_LR', type=float,  default=1e-5)

# model
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing."
                                                                        "No smoothing if None or 0")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='ViT-B-16')
parser.add_argument('--resnet50_pretrain', type=str, default='places_moco',
                        help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
parser.add_argument('--feat_dim', type=int, default=128, help="Feature vector dim, only for classifier32 at the moment")

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=None)

# misc
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--split_train_val', default=False, type=str2bool,
                        help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--use_default_parameters', default=False, type=str2bool,
                    help='Set to True to use optimized hyper-parameters from paper', metavar='BOOL')
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')
parser.add_argument('--gpus', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--checkpt_freq', type=int, default=20)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
parser.add_argument('--train_feat_extractor', default=True, type=str2bool,
                        help='Train feature extractor (only implemented for renset_50_faces)', metavar='BOOL')
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--use_softmax_in_eval', default=False, type=str2bool,
                        help='Do we use softmax or logits for evaluation', metavar='BOOL')

parser.add_argument('--INIT_WEIGHTS', default=False, type=str2bool)
#Prompt
parser.add_argument('--N_CTX', type=int, default=4)
parser.add_argument('--CTX_INIT', type=str, default="a photo of a")

parser.add_argument('--ctp', type=str, default='end')
parser.add_argument('--backbone', type=str, default='ViT-B-16')
parser.add_argument('--prec', type=str, default='fp32')  # fp16, fp32, amp
parser.add_argument('--method', type=str, default='cocoop')
parser.add_argument('--oriweight', type=float, default=1.0, help="weight for features of original images")
parser.add_argument('--clipweight', type=float, default=1.0, help="weight for features of CLIP images")

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def get_optimizer(args, params_list):
    if args.optim is None:
        if args.dataset == 'tinyimagenet':
            optimizer = torch.optim.Adam(params_list, lr=args.lr)
        else:
            optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(params_list, lr=args.lr)

    else:
        raise NotImplementedError

    return optimizer


def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()


def main_worker(args, log):
    # -----------------------------
    # DATALOADERS
    # -----------------------------
    trainloader = dataloaders['train']
    testloader = dataloaders['val']
    outloader = dataloaders['test_unknown']
    # -----------------------------
    # MODEL
    # -----------------------------
    log.info("Creating model: {}".format(args.method))
    if args.method == 'cocoop':
        from models.models_cocoop import get_cocoop_model
    elif args.method == 'cocoop_cat':
        from models.models_cocoop_cat import get_cocoop_model
    elif args.method == 'cocoop_swin_qkv_res':
        from models.models_cocoop_swin_qkv_res import get_cocoop_model
    elif args.method == 'cocoop_swin_layernorm_res':
        from models.models_cocoop_swin_layernorm_res import get_cocoop_model
    elif args.method == 'cocoop_cat_cross_layernorm_res':
        from models.models_cocoop_cat_cross_layernorm_res import get_cocoop_model
    elif args.method == 'cocoop_cat_cross_layernorm_res_lstm':
        from models.models_cocoop_cat_cross_layernorm_res_lstm import get_cocoop_model
    elif args.method == 'cocoop_cat_cross_layernorm_res_2meta':
        from models.models_cocoop_cat_cross_layernorm_res_2meta import get_cocoop_model
    elif args.method =='cocoop_tinyvit':
        from models.models_cocoop_tinyvit import get_cocoop_model
    elif args.method == 'cocoop_cat_2meta':
        from models.models_cocoop_cat_2meta import get_cocoop_model
    elif args.method == 'cocoop_cat_noscale':
        from models.models_cocoop_cat_noscale import get_cocoop_model
    elif args.method == 'coop':
        from models.models_coop import get_cocoop_model
    elif args.method == 'clip':
        from models.models_clip import get_cocoop_model

    net = get_cocoop_model(args, log)

    net.to('cuda')
    # NOTE: only give prompt_learner to the optimizer
    optimizer = build_optimizer(net.prompt_learner, args)
    sched = build_lr_scheduler(optimizer, args)
    # net = nn.DataParallel(net)
    scaler = GradScaler() if args.prec == "amp" else None
    # -----------------------------
    # GET SCHEDULER
    # ----------------------------
    # scheduler = get_scheduler(optimizer, args)

    start_time = time.time()

    # -----------------------------
    # TRAIN
    # -----------------------------
    results = []
    for epoch in range(args.MAX_EPOCH):
        log.info("==> Epoch {}/{}".format(epoch+1, args.MAX_EPOCH))

        train(net, optimizer, trainloader, epoch, args, log)
        sched.step()
        if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.MAX_EPOCH:
            print("==> Test", args.loss)
            result_epoch = test(net, testloader, outloader, epoch, args, log)
            results.append(result_epoch)
            log.info("Epoch {}: Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(epoch,
                                                                                              result_epoch['ACC'],
                                                                                              result_epoch['AUROC'],
                                                                                              result_epoch['OSCR']))
            if epoch % args.checkpt_freq == 0 or epoch == args.MAX_EPOCH - 1:
                weights = net.state_dict()
                result_dir = file_name.split('.')[0]+'_{}'.format(epoch)
                # filename = '{}/{}.pth'.format(args.chkpt_dir, result_dir)
                # torch.save(weights, filename)     # save checkpoints
            args.writer.add_scalar('Test Acc Top 1', result_epoch['ACC'], epoch)
            args.writer.add_scalar('AUROC', result_epoch['AUROC'], epoch)

        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    log.info("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    return results

if __name__ == '__main__':
    args = parser.parse_args()
    # ------------------------
    # Update parameters with default hyperparameters if specified
    # ------------------------
    if args.use_default_parameters:
        print('NOTE: Using default hyper-parameters...')
        args = get_default_hyperparameters(args)

    args.exp_root = exp_root
    args.epochs = args.MAX_EPOCH
    args.eval_freq = 1
    img_size = args.image_size

    torch.manual_seed(args.seed)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.use_gpu = torch.cuda.is_available()
    if args.use_cpu: args.use_gpu = False

    if args.use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    results = dict()
    for i in range(1):
        # ------------------------
        # INIT
        # ------------------------
        if args.feat_dim is None:
            args.feat_dim = 128 if args.model == 'classifier32' else 2048

        args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx,
                                                                     cifar_plus_n=args.out_num)

        args = init_experiment(args)  # chkpt_dir, writer, log_path
        log = Log(__name__, args.log_path).getlog()
        # ------------------------
        # SEED
        # ------------------------
        seed_torch(args.seed)
        # ------------------------
        # DATASETS
        # ------------------------
        datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes, balance_open_set_eval=True,
                                split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                                args=args)
        # ------------------------
        # RANDAUG HYPERPARAM SWEEP
        # ------------------------
        if args.transform == 'rand-augment':
            if args.rand_aug_m is not None:
                if args.rand_aug_n is not None:
                    datasets['train'].transform.transforms[0].m = args.rand_aug_m
                    datasets['train'].transform.transforms[0].n = args.rand_aug_n

        target_classes = np.array(datasets['train'].classes)[args.train_classes].tolist()
        # ------------------------
        # TARGET CLASSES names
        # ------------------------
        if args.dataset in ['tinyimagenet']:
            word_txt = '/home/xyf/dataset/OSR/tiny-imagenet-200/words.txt'
            target_class_tmp = []
            words_dict = {}
            words_name = pd.read_csv(word_txt, names=['indexs', 'name'], header=None, sep="\t")
            for i in words_name.index:
                words = words_name.loc[i].values
                words_dict[words[0]] = words[1]

            for i in target_classes:
                target_class_tmp.append(words_dict[i])
            target_classes = target_class_tmp

        args.target_classes = target_classes
        log.info(target_classes)
        # ------------------------
        # DATALOADER
        # ------------------------
        dataloaders = {}
        for k, v, in datasets.items():
            shuffle = True if k == 'train' else False
            dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                        shuffle=shuffle, sampler=None, num_workers=args.num_workers)
        # ------------------------
        if args.dataset == 'cifar-10-100':
            file_name = '{}-{}.csv'.format(args.dataset, args.out_num)
            if args.cs:
                file_name = '{}_{}_cs.csv'.format(args.dataset, args.out_num)
        else:
            file_name = args.dataset + '.csv'
            if args.cs:
                file_name = args.dataset + 'cs' + '.csv'
        # ------------------------
        # TRAIN
        # ------------------------
        results = main_worker(args, log)
        df = pd.DataFrame(results)
        columns = ['EPOCH', 'ACC', 'AUROC', 'OSCR', 'TNR', 'DTACC', 'AUIN', 'AUOUT', 'AUPR']   # reorder columns
        df = df[columns]
        df.to_csv(os.path.join(args.train_dir, file_name), mode='a', header=True, index=False)