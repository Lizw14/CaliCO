
import argparse
#import tensorboard_logger as tb_logger
import os.path as osp
from collections import namedtuple
from tqdm import tqdm
import time
import json

import torch
from torch.utils.tensorboard import SummaryWriter

from core.optimization import get_linear_schedule_with_warmup
from core.datasets.datasets import build_gqa_dataset, build_gqa_dataloader
from core.models.model import build_model
from core.datasets.defi_attrcept import load_gdef

import pdb
import os

parser = argparse.ArgumentParser()

# dataset path
parser.add_argument('--train_question_json_path', default='data/orig_data/questions1.2/train_balanced_questions.json')
parser.add_argument('--val_question_json_path', default='data/orig_data/questions1.2/testdev_balanced_questions.json')
parser.add_argument('--info_json_path', default='data/orig_data/merged_features/gqa_objects_merged_info.json')
parser.add_argument('--objects_h5_path', default='data/orig_data/merged_features/gqa_objects.h5')
parser.add_argument('--train_scenegraph_json_path', default='data/orig_data/sceneGraphs/train_sceneGraphs.json')
parser.add_argument('--val_scenegraph_json_path', default='data/orig_data/sceneGraphs/val_sceneGraphs.json')
parser.add_argument('--vocab_json_path', default='data/gqa_vocab_13.json')
parser.add_argument('--dataset_cache', default=None)
parser.add_argument('--test_split', default='testdev')
parser.add_argument('--is_gtencode', type=bool, default=False)

# exp
parser.add_argument('--ckpt_path')
parser.add_argument('--pretrained_path', default=None)
parser.add_argument('--eval_every', type=int, default=2, help='int, eval every how many epoches')
parser.add_argument('--ckpt_every', type=int, default=2)
parser.add_argument('--log_every', type=int, default=20)
parser.add_argument('--is_test', type=bool, default=False, help='training or testing mode')

# training
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_worker', type=int, default=0, help='number of dataset workers')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--num_train_epoches', type=int, default=50)
parser.add_argument('--warmup_steps', type=int, default=4000)

# model
parser.add_argument('--gdef_pth', default='data/define_prog_12.json')
parser.add_argument('--sgg_vocab_pth', default='/home/zhuowan/zhuowan/gqa_project/Scene-Graph-Benchmark.pytorch/datasets/GQA/gqa_vocab_thres1.json')
parser.add_argument('--input_dims', metavar='N', type=int, nargs='+', default=[2048,2048])
parser.add_argument('--hidden_dims', metavar='N', type=int, nargs='+', default=[256,256])
parser.add_argument('--if_hier', type=bool, default=False, help='whether to use hierarchical supervision')
args = parser.parse_args()

class Trainer():
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.history_val_result = {'iters':[], 'acc': []}
        self.history_train_result = {'iters':[], 'acc': []}
        self.best_val_result = -1
        self.best_val_iter = -1

    def train(self):
        # build dataloaders
        if not args.is_test:
            train_dataset_args = namedtuple("ARGS",
                ['question_json_path', 'info_json_path', 'objects_h5_path', 'scenegraphs_json_path', 'vocab_json_path',
                'mode', 'dataset_cache', 'is_gtencode', 'data_num'])(args.train_question_json_path, args.info_json_path, args.objects_h5_path, args.train_scenegraph_json_path, args.vocab_json_path,
                'train', args.dataset_cache, args.is_gtencode, None)
            train_dataset = build_gqa_dataset(train_dataset_args)
            self.train_loader = build_gqa_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_worker)
        self.val_loaders = {}
        data_num = None if args.is_test else 20000
        for question_json_path, scenegraph_json_path in zip(args.val_question_json_path.split(','), args.val_scenegraph_json_path.split(',')):
            val_dataset_args = namedtuple("ARGS",
                ['question_json_path', 'info_json_path', 'objects_h5_path', 'scenegraphs_json_path', 'vocab_json_path',
                'mode', 'dataset_cache', 'is_gtencode', 'data_num'])(question_json_path, args.info_json_path, args.objects_h5_path, scenegraph_json_path, args.vocab_json_path,
                args.test_split, args.dataset_cache, args.is_gtencode, data_num)
            val_dataset = build_gqa_dataset(val_dataset_args)
            val_loader = build_gqa_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_worker)
            self.val_loaders[question_json_path.split('/')[-1].split('_')[0]] = val_loader #testdev, val

        # definition of attributes/relations/operations
        gdef = load_gdef(gdef_pth=args.gdef_pth, sgg_vocab_pth=args.sgg_vocab_pth)

        # build model
        ckpt_pth = None
        if args.is_test:
            ckpt_pth = args.ckpt_path
        if args.pretrained_path is not None:
            ckpt_pth = args.pretrained_path
        model_configs = namedtuple("ARGS", 
            ['input_dims', 'hidden_dims', 'if_hier'])(args.input_dims, args.hidden_dims, args.if_hier)
        if args.is_gtencode:
            model_configs.input_dims[0] = len(val_dataset.unwrapped.sg_all_concepts)
            model_configs.input_dims[1] = len(val_dataset.unwrapped.sg_all_concepts)
        self.model = build_model(model_configs, gdef, ckpt_pth)
        self.model.cuda()

        # test case
        if args.is_test:
            acc = self.eval_epoch(0, verbose=True)
            print('Acc: ', acc)
            return 0

        # optimizer
        params = self.model.parameters()
        self.optimizer = torch.optim.Adam(params, lr=args.lr)

	    # scheduler
        t_total = len(self.train_loader) * args.num_train_epoches
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # tb_logger
        self.summary_writer = SummaryWriter(osp.join(args.ckpt_path, 'my_logger'))

        # training epoches
        start_iter = 1
        for epoch in range(1, args.num_train_epoches+1):
            print('Starting training %d epoches' % epoch)
            acc_dict = self.train_epoch(start_iter)
            print('Finished training %d epoches' % epoch)
            if epoch % args.eval_every == 0:
                print(args.ckpt_path + ': Evaluating for %d epoch ... ' % epoch)
                val_accs = self.eval_epoch(start_iter)
                acc_dict.update(val_accs)
                print('Acc after epoch %d: ' % (epoch))
                print(acc_dict)
                self.summary_writer.add_scalars('Acc/acc', acc_dict, start_iter)
            if epoch % args.ckpt_every == 0:
                checkpoint = {
                    'model': self.model.state_dict(),
                    'args': self.args,
                    'iters': start_iter,
                    'best_val_iter': self.best_val_iter,
                    'best_val_result': self.best_val_result,
                    'history_val_result': self.history_val_result,
                    'history_train_result': self.history_train_result
                }
                print('Saving checkpoint ...')
                self.model.save_ckpt(checkpoint, args.ckpt_path)
            start_iter += len(self.train_loader)


    def train_epoch(self, start_iter):
        is_corrects, types = [], []

        epoch = start_iter // len(self.train_loader)
        time_start = time.time()
        for idx, batch in enumerate(self.train_loader):
            current_iter = start_iter + idx
            loss, misc = self.model(batch)
            if current_iter % args.log_every == 0:
                self.summary_writer.add_scalar('Loss/loss', loss.item(), current_iter)
                self.summary_writer.add_scalar('Optim/lr', self.scheduler.get_lr()[0], current_iter)
                for logname in misc[0]['loss_type']: #misc[0]['loss_type'] is a dict of dicts
                    log_value = {op:v for op,v in misc[0]['loss_type'][logname].items() if v is not None}
                    self.summary_writer.add_scalars('Loss/loss_'+logname, log_value, current_iter)
                print('Epoch %d, iter %d/%d, loss %.2f, spending %.1fs, acc %.2f.' % (epoch, current_iter%len(self.train_loader), len(self.train_loader), loss, time.time()-time_start, torch.tensor(misc[0]['is_correct']).float().mean().item()))
                time_start = time.time()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            with torch.no_grad():
                types.extend(batch['question_type'])
                is_corrects.extend(misc[0]['is_correct'])

        # log
        acc = torch.tensor(is_corrects).float().mean().item()
        type_accs = {}
        for idx, qtype in enumerate(types):
            if qtype not in type_accs:
                type_accs[qtype] = []
            type_accs[qtype].append(is_corrects[idx])
        print('Training accs: ')
        print('Acc: ', acc)
        for qtype in type_accs:
            print(qtype, ': ', len(type_accs[qtype]), ', ', torch.tensor(type_accs[qtype]).float().mean().item())

        # update trainer record
        end_iter = start_iter+len(self.train_loader)
        self.history_train_result['iters'].append(end_iter)
        self.history_train_result['acc'].append(acc)
        log_dict = {qtype: torch.tensor(type_accs[qtype]).float().mean().item() for qtype in type_accs}
        self.summary_writer.add_scalars('Acc/acc_type_train', log_dict, end_iter)
        return {'train': acc}


    def eval_epoch(self, start_iter, verbose=False):
        self.model.eval()
        acc_dict = {}
        for val_loader_name, val_loader in self.val_loaders.items():
            is_corrects = []
            types = []
            if verbose:
                preds = []
                gts = []
                questions = []
                img_ids = []
                questionIds = []
                weights = []

            with torch.no_grad():
                for batch in tqdm(val_loader):
                    # run forward and compute acc
                    _, output = self.model(batch)
                    is_corrects.extend(output[0]['is_correct'])
                    types.extend(batch['question_type'])
                    if verbose:
                        preds.extend(output[0]['answer'])
                        gts.extend(batch['answer_raw'])
                        questions.extend(batch['question_raw'])
                        img_ids.extend(batch['imageId'])
                        questionIds.extend(batch['qid'])

            acc = torch.tensor(is_corrects).float().mean().item()
            type_accs = {}
            for idx, qtype in enumerate(types):
                if qtype not in type_accs:
                    type_accs[qtype] = []
                type_accs[qtype].append(is_corrects[idx])
            print('Evaluation on split '+val_loader_name)
            print('Acc: ', acc)
            for qtype in type_accs:
                print(qtype, ': ', len(type_accs[qtype]), ', ', torch.tensor(type_accs[qtype]).float().mean().item())

            if verbose:
                with open(osp.join(args.ckpt_path, val_loader_name+'_result.json'), 'w') as f:
                    results = {}
                    for is_correct, pred, gt, question, img_id, questionId in zip(is_corrects, preds, gts, questions, img_ids, questionIds):
                        results[questionId] = {'prediction': pred}
                    json.dump(results, f)
                with open(osp.join(args.ckpt_path, val_loader_name+'_result.txt'), 'w') as f:
                    for is_correct, pred, gt, question, img_id in zip(is_corrects, preds, gts, questions, img_ids):
                        f.write(str(is_correct)+', '+str(pred)+', '+str(gt)+', '+str(question)+ ', '+str(img_id)+'\n')

            # update trainer record
            if not args.is_test:
                if val_loader_name=='testdev':
                    self.history_val_result['iters'].append(start_iter)
                    self.history_val_result['acc'].append(acc)
                    if acc >= self.best_val_result:   
                        self.best_val_result = acc
                        self.best_val_iter = start_iter
                acc_dict[val_loader_name] = acc
                log_dict = {qtype: torch.tensor(type_accs[qtype]).float().mean().item() for qtype in type_accs}
                self.summary_writer.add_scalars('Acc/'+val_loader_name+'_acc_type', log_dict, start_iter)

        self.model.train()
        return acc_dict



if __name__ == "__main__":
    trainer = Trainer(args)
    trainer.train()
