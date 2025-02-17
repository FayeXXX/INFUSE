import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm
from methods.ARPL.core import evaluation

from sklearn.metrics import average_precision_score, f1_score

def test(net, testloader, outloader, args):
    threshold = 0.5
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels, predict_u = [], [], [], []

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

                # for F1 score
                predict_logits = torch.nn.Softmax(dim=-1)(logits)
                maxlogits = torch.max(predict_logits, dim=1)[0]
                prediction = torch.zeros([data.shape[0]], requires_grad=False).to('cuda')
                for i in range(data.shape[0]):
                    if maxlogits[i] < threshold:
                        prediction[i] = -1
                predict_u.append(prediction.data.cpu().numpy())


                _pred_u.append(logits.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    # log.info('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)

#   save the results for ood experiments
    project_root_dir = '/home/xyf/PycharmProjects/osr_closed_set_all_you_need/'
    save_path = f'{project_root_dir}/ood_experiments/{args.dataset}/{args.method}/{args.backbone}/epoch0'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(f'{save_path}/_pred_k.npy', _pred_k)
    np.save(f'{save_path}/_pred_u.npy', _pred_u)
    np.save(f'{save_path}/_labels.npy', _labels)
    # Out-of-Distribution detction evaluation
