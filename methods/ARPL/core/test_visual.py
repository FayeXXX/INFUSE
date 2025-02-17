import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm
from methods.ARPL.core import evaluation

from sklearn.metrics import average_precision_score

from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import seaborn as sns
import torch.nn.functional as F

def subsample_classes(dataset, include_classes=range(20)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    return dataset


def get_image_paths_from_dataloader(dataloader):
    # 获取 Dataset 对象
    dataset = dataloader.dataset

    # 提取所有图像路径
    image_paths, label = [img_path for img_path, label in dataset.samples]

    return image_paths, label


def plot_distribution(id_scores, ood_scores, out_dataset, args):
    sns.set(style="white", palette="muted")
    palette = ['#A8BAE3', '#55AB83']
    sns.displot({"ID":id_scores, "OOD": ood_scores}, label="id", kind = "kde", palette=palette, fill = True, alpha = 0.8)
    plt.savefig(f"ood_figs/{args.dataset}_split:{args.split_idx}.pdf", bbox_inches='tight')


def test(net, testloader, outloader, args, log):
    log.info("test begin")
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels_k, _labels_u = [], [], [], []

    dataset_k = testloader.dataset
    dataset_u = outloader.dataset
    infor_k, infor_u = [], []

    with torch.no_grad():
        for batch_idx, (data, labels, idx) in tqdm(enumerate(testloader)):

            data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                logits = net(data, True)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()

                # if args.use_softmax_in_eval:
                logits = torch.nn.Softmax(dim=-1)(logits)

                _pred_k.append(logits.data.cpu().numpy())
                _labels_k.append(labels.data.cpu().numpy())

                # get path and label as infor_k
                # bs = testloader.batch_size
                # start = batch_idx * bs
                # end = min(start + bs, len(dataset_k.samples))
                # infor_k.append(dataset_k.samples[start:end])

        for batch_idx, (data, labels, idx) in enumerate(tqdm(outloader)):

            data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):

                logits = net(data, None)

                # if args.use_softmax_in_eval:
                logits = torch.nn.Softmax(dim=-1)(logits)

                _pred_u.append(logits.data.cpu().numpy())
                _labels_u.append(labels.data.cpu().numpy())

                # get path and label as infor_u
                bs = outloader.batch_size
                start = batch_idx * bs
                end = min(start + bs, len(dataset_u.samples))
                infor_u.append(dataset_u.samples[start:end])

    # Accuracy
    # acc = float(correct) * 100. / float(total)_labels.append(labels.data.cpu().numpy())
    # log.info('Acc: {:.5f}'.format(acc))
    log.info("test done")  # value on current batch, average value

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _pred_k, _pred_u = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    _pred_k_label, _pred_u_label = np.argmax(_pred_k, axis=1), np.argmax(_pred_u, axis=1)

    df_k = pd.DataFrame({'Labels': np.concatenate(_labels_k, 0), 'Scores': np.concatenate([_pred_k], 0), \
                        'Path': np.concatenate(infor_k, 0)[:, 0], 'Origin_label':np.concatenate(infor_k, 0)[:, 1]})
    csv_file_k = 'ood_figs/' + args.method+'_'+args.dataset + '_split:' + str(args.split_idx) + '_known'+ '.csv'
    df_k.to_csv(csv_file_k, index=False)

    df_u = pd.DataFrame({'Labels': np.concatenate(_labels_u, 0), 'Scores': np.concatenate([_pred_u], 0),\
                         'Path': np.concatenate(infor_u, 0)[:, 0], 'Origin_label':np.concatenate(infor_u, 0)[:, 1]})
    csv_file_u = 'ood_figs/' + args.method+'_'+ args.dataset + '_split:' + str(args.split_idx) + '_unknown'+'.csv'
    df_u.to_csv(csv_file_u, index=False)

    #

    plot_distribution(_pred_k, _pred_u, 'imagenet', args)



    return




