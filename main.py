import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train, test
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--train', type=bool, default=True, help='True/False')
    parser.add_argument('--modelpath', type=str, default='saved_models/exp0/model.pth')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', None, dictionary)
    eval_dset = VQAFeatureDataset('val', None, dictionary)

    batch_size = args.batch_size
    constructor = 'build_%s' % args.model

    if args.train:    

        model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda() 
        model.w_emb.init_embedding('data/glove6b_init_300d.npy')
        model = nn.DataParallel(model).cuda()

        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
        eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
        train(model, train_loader, eval_loader, args.epochs, args.output)


    else:

        teststd_dset = VQAFeatureDataset('val', 'std', dictionary)
        testdev_dset = VQAFeatureDataset('val', 'dev', dictionary)
        
        model = getattr(base_model, constructor)(teststd_dset, args.num_hid).cuda()
        model.w_emb.init_embedding('data/glove6b_init_300d.npy')
        model.load_state_dict(torch.load(args.modelpath))
        
        teststd_loader =  DataLoader(teststd_dset, batch_size, shuffle=True, num_workers=1)
        testdev_loader =  DataLoader(teststd_dset, batch_size, shuffle=True, num_workers=1)
        test(model, teststd_loader, teststd_dset.label2ans)
        test(model, testdev_loader, testdev_dset.label2ans)