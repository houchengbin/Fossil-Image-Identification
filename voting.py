#!/usr/bin/env python3


import time
import argparse
import logging
import numpy as np
import torch
from timm.models import create_model, apply_test_time_pool, load_checkpoint
from timm.data import ImageDataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging, accuracy
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


torch.backends.cudnn.benchmark = True

_logger = logging.getLogger('voting')
parser = argparse.ArgumentParser(description='voting log')

parser.add_argument('--view1', '-v1', metavar='DIR',
                    help='path to view1 dataset')
parser.add_argument('--view2', '-v2', metavar='DIR',
                    help='path to view2 dataset')
parser.add_argument('--view3', '-v3', metavar='DIR',
                    help='path to view3 dataset')

parser.add_argument('--model-view1', '-m1', '--model', metavar='MODEL', default=None,
                    help='model architecture for view1  (default: none)')
parser.add_argument('--model-view2', '-m2', metavar='MODEL', default=None,
                    help='model architecture for view2  (default: none)')
parser.add_argument('--model-view3', '-m3', metavar='MODEL', default=None,
                    help='model architecture for view3  (default: none)')

parser.add_argument('--checkpoint-view1', '-cp1', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint for view1 (default: none)')
parser.add_argument('--checkpoint-view2', '-cp2', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint for view2 (default: none)')
parser.add_argument('--checkpoint-view3', '-cp3', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint for view3 (default: none)')

parser.add_argument('--num-classes', type=int, default=16,
                    help='Number classes in dataset')

# timm default parameters
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=5, type=int,
                    metavar='N', help='Top-k to output to CSV')


def validate(loader_view, model, args):
    with torch.no_grad():
    
        top1_view = AverageMeter()
        top3_view = AverageMeter()
        top5_view = AverageMeter()
        top10_view = AverageMeter()

        batch_time = AverageMeter()
        end = time.time()
        
        for batch_idx, (input, target) in enumerate(loader_view):
            input = input.cuda()
            
            target = target.cuda()
            scores_view = model(input)
            
            if batch_idx == 0:
              scores = scores_view
              target_ids = target
              
            if batch_idx > 0:
              scores = torch.cat((scores,scores_view),dim=0)
              target_ids = torch.cat((target_ids,target),dim=0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # measure accuracy
            acc1, acc3, acc5, acc10 = accuracy(scores_view.detach(), target, topk=(1, 3, 5, 10))

            top1_view.update(acc1.item(), input.size(0))
            top3_view.update(acc3.item(), input.size(0))
            top5_view.update(acc5.item(), input.size(0))
            top10_view.update(acc10.item(), input.size(0))

            if batch_idx % args.log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader_view), batch_time=batch_time))
                
        return scores, target_ids, top1_view, top3_view, top5_view, top10_view

# soft-voting
def average(outputs):
    return sum(outputs) / len(outputs)

# hard-voting
def majority_vote(outputs: list[torch.Tensor]) -> torch.Tensor:
    '''
    Compute the majority vote for a list of model outputs.
    outputs: list of length (n_models)
    containing tensors with shape (n_samples, n_classes)
    majority_one_hots: (n_samples, n_classes)
    '''

    if len(outputs[0].shape) != 2:
        msg = """The shape of outputs should be a list tensors of
        length (n_models) with sizes (n_samples, n_classes).
        The first tensor had shape {} """
        raise ValueError(msg.format(outputs[0].shape))
  
    votes = torch.stack(outputs).argmax(dim=2).mode(dim=0)[0]
    proba = torch.zeros_like(outputs[0])
    majority_one_hots = proba.scatter_(1, votes.view(-1, 1), 1)
  
    return majority_one_hots


def main():
    setup_default_logging()
    args = parser.parse_args()
    
    if args.model_view2 == None and args.model_view3 == None:
        args.model_view2 = args.model_view1
        args.model_view3 = args.model_view1


    # create model - 3 views
    model_view1 = create_model(
        args.model_view1,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained)
        
    if args.checkpoint_view1:
        load_checkpoint(model_view1, args.checkpoint_view1, strict=False)
        
    model_view2 = create_model(
        args.model_view2,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained)
    
    if args.checkpoint_view2:
        load_checkpoint(model_view2, args.checkpoint_view2, strict=False)
        
    model_view3 = create_model(
        args.model_view3,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained)
        
    if args.checkpoint_view3:
        load_checkpoint(model_view3, args.checkpoint_view3, strict=False)    
    


    _logger.info('Model_view1 %s created, param count: %d' %
                 (args.model_view1, sum([m.numel() for m in model_view1.parameters()])))
    _logger.info('Model_view2 %s created, param count: %d' %
                 (args.model_view2, sum([m.numel() for m in model_view2.parameters()])))
    _logger.info('Model_view3 %s created, param count: %d' %
                 (args.model_view3, sum([m.numel() for m in model_view3.parameters()])))


    config_view1 = resolve_data_config(vars(args), model=model_view1)
    config_view2 = resolve_data_config(vars(args), model=model_view2)
    config_view3 = resolve_data_config(vars(args), model=model_view3)
    
    model_view1, test_time_pool = (model_view1, False) if args.no_test_pool else apply_test_time_pool(model_view1, config_view1)
    model_view2, test_time_pool = (model_view2, False) if args.no_test_pool else apply_test_time_pool(model_view2, config_view2)
    model_view3, test_time_pool = (model_view3, False) if args.no_test_pool else apply_test_time_pool(model_view3, config_view3)


    if args.num_gpu > 1:
        model_view1 = torch.nn.DataParallel(model_view1, device_ids=list(range(args.num_gpu))).cuda()
        model_view2 = torch.nn.DataParallel(model_view2, device_ids=list(range(args.num_gpu))).cuda()
        model_view3 = torch.nn.DataParallel(model_view3, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model_view1 = model_view1.cuda()
        model_view2 = model_view2.cuda()
        model_view3 = model_view3.cuda()

    loader_view1 = create_loader(
        ImageDataset(args.view1),
        input_size=config_view1['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=config_view1['interpolation'],
        mean=config_view1['mean'],
        std=config_view1['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else config_view1['crop_pct'])
    
    loader_view2 = create_loader(
        ImageDataset(args.view2),
        input_size=config_view2['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=config_view2['interpolation'],
        mean=config_view2['mean'],
        std=config_view2['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else config_view2['crop_pct'])
        
    loader_view3 = create_loader(
        ImageDataset(args.view3),
        input_size=config_view3['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=config_view3['interpolation'],
        mean=config_view3['mean'],
        std=config_view3['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else config_view3['crop_pct'])
    
    model_view1.eval()
    model_view2.eval()
    model_view3.eval()
    
    
    print('--------------------------------------------------------------------------------------')
    # ------------ view 1 ----------------- image
    scores_view1, target, top1_view1, top3_view1, top5_view1, top10_view1 = validate(loader_view1, model_view1, args)
    print('--- model_view1-image: ','|','TestACC@1:',top1_view1.avg,'|','|','ACC@3:',top3_view1.avg,'|','|','ACC@5:',top5_view1.avg,'|','|','ACC@10:',top10_view1.avg,'|')
    # ------------ view 2 ----------------- binary
    scores_view2, target_nouse, top1_view2, top3_view2, top5_view2, top10_view2 = validate(loader_view2, model_view2, args)
    print('--- model_view2-image: ','|','TestACC@1:',top1_view2.avg,'|','|','ACC@3:',top3_view2.avg,'|','|','ACC@5:',top5_view2.avg,'|','|','ACC@10:',top10_view2.avg,'|')
    # ------------ view 3 ----------------- skeleton
    scores_view3, target_nouse, top1_view3, top3_view3, top5_view3, top10_view3 = validate(loader_view3, model_view3, args)
    print('--- model_view3-image: ','|','TestACC@1:',top1_view3.avg,'|','|','ACC@3:',top3_view3.avg,'|','|','ACC@5:',top5_view3.avg,'|','|','ACC@10:',top10_view3.avg,'|')

    # ----- soft ---------
    print('--------------------------------------------------------------------------------------')
    prob_all = []
    prob_all.append(torch.nn.functional.softmax(scores_view1, dim=1))
    prob_all.append(torch.nn.functional.softmax(scores_view2, dim=1))
    prob_all.append(torch.nn.functional.softmax(scores_view3, dim=1))
    soft = average(prob_all)
    acc1, acc3, acc5, acc10 = accuracy(soft.detach(), target, topk=(1, 3, 5, 10))
    print('--- model_3view softvoting: ','|','TestACC@1:',acc1.cpu().numpy(),'|','|','ACC@3:',acc3.cpu().numpy(),'|','|','ACC@5:',acc5.cpu().numpy(),'|','|','ACC@10:',acc10.cpu().numpy(),'|')
    
    # ----- hard -------
    print('--------------------------------------------------------------------------------------')
    
    hard = majority_vote(prob_all)
    acc1, acc3, acc5, acc10 = accuracy(hard.detach(), target, topk=(1, 3, 5, 10))
    
    print('--- model_3view hardvoting: ','|','TestACC@1:',acc1.cpu().numpy(),'|','|','ACC@3:',acc3.cpu().numpy(),'|','|','ACC@5:',acc5.cpu().numpy(),'|','|','ACC@10:',acc10.cpu().numpy(),'|')
    print('--------------------------------------------------------------------------------------')
    
    # ------precision recall F1 -------
    
    voting_pred = np.concatenate(soft.topk(1)[1].cpu().numpy(),axis=0).tolist()
    voting_true = target.cpu().numpy().tolist()
    
    #print('--------------------------------------------------------------------------------------')
    #print(classification_report(voting_true, voting_pred, digits=6))
    #print('--------------------------------------------------------------------------------------')
    print("sklearn accuracy: ", accuracy_score(voting_true, voting_pred))
    print("precision_score_macro: ", precision_score(voting_true, voting_pred, average='macro'))
    print("precision_score_micro: ", precision_score(voting_true, voting_pred, average='micro'))
    print("recall_score_macro: ", recall_score(voting_true, voting_pred, average='macro'))
    print("recall_score_micro: ", recall_score(voting_true, voting_pred, average='micro'))
    print("f1_score_macro: ", f1_score(voting_true, voting_pred, average='macro'))
    print("f1_score_micro: ", f1_score(voting_true, voting_pred, average='micro'))
    print('--------------------------------------------------------------------------------------')

    

if __name__ == '__main__':
    main()

