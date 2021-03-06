#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
from automated_evaluation.main import imdb
from automated_evaluation import model
import torch.autograd as autograd

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=1, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=True, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

# load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)

train_iter, dev_iter, test_iter = imdb(text_field, label_field, device=-1, repeat=False)
# train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
# train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)


# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available()
del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
cnn = model.CNN_Text(args)
snapshot = 'automated_evaluation/best_steps_11513.pt'
if snapshot is not None:
    print('\nLoading model from {}...'.format(snapshot))
    cnn.load_state_dict(torch.load(snapshot, map_location=torch.device('cpu')))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()


def sent_acc(samples, model, text_field, cuda_flag, positive=True):
    size = len(samples)
    model.eval()
    # text = text_field.tokenize(text)
    outputs = torch.tensor([], dtype=torch.int64)
    for sample in samples:
        sample = text_field.preprocess(sample)
        sample = [[text_field.vocab.stoi[x] for x in sample]]
        # inputs.append(sample)
        x = torch.tensor(sample)
        x = autograd.Variable(x)
        if cuda_flag:
            x = x.cuda()
        # print(x)
        output = model(x)
        _, predicted = torch.max(output, 1)  # logits
        outputs = torch.cat([outputs, predicted])

    target = [1] * size if positive else [0] * size
    target = torch.tensor(target)
    corrects = outputs == target
    corrects = corrects.sum()
    accuracy = 100.0 * corrects / size
    # return label_feild.vocab.itos[predicted.data[0][0]+1]
    return accuracy


def cal_acc(samples, pos=True):
    acc = sent_acc(samples, cnn, text_field, args.cuda, pos)
    return acc


def cal_accs(method_label, suffix, src):
    print('{}:\n'.format(method_label))
    pos_acc = cal_acc('positive', method_label, suffix, src)
    print('pos_acc: {}'.format(pos_acc.item()))

    neg_acc = cal_acc('negative', method_label, suffix, src)
    print('neg_acc: {}'.format(neg_acc.item()))

    print('mean_acc: {}'.format((pos_acc + neg_acc) / 2))


def test_num_samples(method_label, suffix, src):
    src_pos = '{}/{}/{}{}'.format(src, 'positive', method_label, suffix)
    src_neg = '{}/{}/{}{}'.format(src, 'negative', method_label, suffix)
    with open(src_pos, 'r') as f:
        samples = f.read().split('<|endoftext|>')
        samples = samples[1:]
        samples = [s for s in samples if len(s.split()) > 10]
        print('num_pos_samples: {}'.format(len(samples)))

    with open(src_neg, 'r') as f:
        samples = f.read().split('<|endoftext|>')
        samples = samples[1:]
        samples = [s for s in samples if len(s.split()) > 10]
        print('num_neg_samples: {}'.format(len(samples)))


# SRC_SAMPLES = '/Users/xuchen/core/pycharm/project/PPL/automated_evaluation/generated_samples'

# # single or not
if __name__ == '__main__':

    snapshot = 'best_steps_11513.pt'
    with open(snapshot, 'r'):
        print('aaa')

    # src = '../data/test/generated_samples/paper'
    # src = '/Users/xuchen/core/pycharm/project/PPL/automated_evaluation/generated_samples/paper'

    # names = [
    #     'bc_neg(2_150_10)',
    #     # 'bc_pos(2_150_10)',
    #     'vad_neg(2_150_10)',
    #     # 'vad_pos(2_150_10)',
    # ]

    # names = [
    #     'bc_neg(2_150_10)',
    #     # 'bc_pos(2_150_10)',
    # ]
    #
    # sent_label = [
    #     # 'positive',
    #     'negative'
    # ]
    #
    # for name in names:
    #     path = '{}/{}'.format(src, name)
    #     acc = cal_acc(path, True)
    #     print('{}: {}'.format(name, acc))
    #
    # # multiple
    # method_labels = [
    #     # 'B',
    #     # 'BC',
    #     'BC_VAD',
    #     # 'BC_VAD_ABS',
    #     # 'BC_VAD_LOSS'
    # ]

    # suffix = '(dsq_2_150_10_0.1)'

    # test_num_samples(method_labels[0], suffix, SRC_SAMPLES)
    # cal_accs(method_labels[0], suffix, SRC_SAMPLES)
    # neg_acc = cal_acc('negative', method_label[0], SRC_SAMPLES)
    # print('neg_acc: {}'.format(neg_acc.item()))
