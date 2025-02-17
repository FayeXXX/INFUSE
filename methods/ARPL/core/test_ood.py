import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm
from methods.ARPL.core import evaluation

from sklearn.metrics import average_precision_score, f1_score

def test(net, testloader, outloader, epoch, args, log):

    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels, idx in tqdm(testloader):
            if args.use_gpu:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                logits = net(data, True)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()

                if args.use_softmax_in_eval:
                    logits = torch.nn.Softmax(dim=-1)(logits)

                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(tqdm(outloader)):
            if args.use_gpu:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):

                logits = net(data, None)

                if args.use_softmax_in_eval:

                    logits = torch.nn.Softmax(dim=-1)(logits)

                _pred_u.append(logits.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    log.info('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)

    # select f1
    F1Score = 0
    for threshold in np.arange(0.5, 1.0, 0.01):
        predict_u, predict_k = [], []
        for logits in _pred_u:
            tensor_logits = torch.from_numpy(logits)
            tensor_logits = tensor_logits.to(torch.float32)
            predict_logits = torch.nn.Softmax(dim=-1)(tensor_logits)
            maxlogits = torch.max(predict_logits)
            if maxlogits < threshold:
                prediction = -1
            else:
                prediction = 0
            predict_u.append(prediction)

        for logits in _pred_k:
            tensor_logits = torch.from_numpy(logits)
            tensor_logits = tensor_logits.to(torch.float32)
            predict_logits = torch.nn.Softmax(dim=-1)(tensor_logits)
            maxlogits = torch.max(predict_logits)
            if maxlogits < threshold:
                prediction = -1
            else:
                prediction = np.argmax(logits, axis=-1)
            predict_k.append(prediction)

        # F1score
        labels_all = np.concatenate([_labels, np.full(10000, -1)], axis=0)
        predictions_all = np.concatenate([predict_k, predict_u], axis=0)
        if f1_score(labels_all, predictions_all, average='macro') > F1Score:
            F1Score = f1_score(labels_all, predictions_all, average='macro')
            # print(f'Threshold: {threshold}, F1 Score: {F1Score}')


    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']

    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    # Average precision
    ap_score = average_precision_score([0] * len(_pred_k) + [1] * len(_pred_u),
                                       list(-np.max(_pred_k, axis=-1)) + list(-np.max(_pred_u, axis=-1)))

    # # F1score
    # labels_all = np.concatenate([_labels, np.full(10000, -1)], axis=0)
    # predict_u = np.concatenate(predict_u, 0)
    # predictions_all = np.concatenate([np.argmax(_pred_k, axis=-1), predict_u], axis=0)
    # F1Score = f1_score(labels_all, predictions_all, average='macro')

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.
    results['AUPR'] = ap_score * 100
    results['EPOCH'] = epoch
    results['F1'] = F1Score * 100

    return results