
import sys
import os
import argparse
import time
import random
import datetime
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboard import summary
from tensorboard import FileWriter
from tqdm import tqdm

from model_factory import get_model_class
from utils import Corpus, EmbeddingLayer, FileLoader, RandomLoader, CombinedLoader, OutputManager
from utils import say, load_embedding, pad_iter, make_batch
from utils.meter import AUCMeter

outputManager = 0

def train(iter_cnt, model, corpus, args, optimizer):

    train_writer = FileWriter(args.run_path+"/train", flush_secs=5)

    pos_file_path = "{}.pos.txt".format(args.train)
    neg_file_path = "{}.neg.txt".format(args.train)
    pos_batch_loader = FileLoader(
        [ tuple([pos_file_path, os.path.dirname(args.train)]) ],
        args.batch_size
    )
    neg_batch_loader = FileLoader(
        [ tuple([neg_file_path, os.path.dirname(args.train)]) ],
        args.batch_size
    )
    #neg_batch_loader = RandomLoader(
    #    corpus = corpus,
    #    exclusive_set = zip(pos_batch_loader.data_left, pos_batch_loader.data_right),
    #    batch_size = args.batch_size
    #)
    #neg_batch_loader = CombinedLoader(
    #    neg_batch_loader_1,
    #    neg_batch_loader_2,
    #    args.batch_size
    #)

    use_content = False
    if args.use_content:
        use_content = True

    embedding_layer = model.embedding_layer

    criterion = model.compute_loss

    start = time.time()
    tot_loss = 0.0
    tot_cnt = 0

    for batch, labels in tqdm(pad_iter(corpus, embedding_layer, pos_batch_loader,
                neg_batch_loader, use_content, pad_left=False)):
        iter_cnt += 1
        model.zero_grad()
        labels = labels.type(torch.LongTensor)
        new_batch = [ ]
        if args.use_content:
            for x in batch:
                for y in x:
                    new_batch.append(y)
            batch = new_batch
        if args.cuda:
            batch = [ x.cuda() for x in batch ]
            labels = labels.cuda()
        batch = map(Variable, batch)
        labels = Variable(labels)
        repr_left = None
        repr_right = None
        if not use_content:
            repr_left = model(batch[0])
            repr_right = model(batch[1])
        else:
            repr_left = model(batch[0]) + model(batch[1])
            repr_right = model(batch[2]) + model(batch[3])
        output = model.compute_similarity(repr_left, repr_right)
        loss = criterion(output, labels)
        loss.backward()
        prev_emb = embedding_layer.embedding.weight.cpu().data.numpy()
        optimizer.step()
        current_emb = embedding_layer.embedding.weight.cpu().data.numpy()
        diff = np.sum(np.absolute(current_emb-prev_emb))
        tot_loss += loss.data[0]*output.size(0)
        tot_cnt += output.size(0)
        if iter_cnt % 100 == 0:
            outputManager.say("\r" + " "*50)
            outputManager.say("\r{} loss: {:.4f}  eps: {:.0f} ".format(
                iter_cnt, tot_loss / tot_cnt,
                tot_cnt/(time.time()-start)
            ))
            s = summary.scalar('loss', tot_loss/tot_cnt)
            train_writer.add_summary(s, iter_cnt)

    outputManager.say("\n")
    train_writer.close()
    #if model.criterion.startswith('classification'):
    #    print model.output_op.weight.min().data[0], model.output_op.weight.max().data[0]

    return iter_cnt

def evaluate(iter_cnt, filepath, model, corpus, args, logging=True):
    if logging:
        valid_writer = FileWriter(args.run_path+"/valid", flush_secs=5)

    pos_file_path = "{}.pos.txt".format(filepath)
    neg_file_path = "{}.neg.txt".format(filepath)
    pos_batch_loader = FileLoader(
        [tuple([pos_file_path, os.path.dirname(args.eval)])],
        args.batch_size
    )
    neg_batch_loader = FileLoader(
        [tuple([neg_file_path, os.path.dirname(args.eval)])],
        args.batch_size
    )


    batchify = lambda bch: make_batch(model.embedding_layer, bch)
    model.eval()
    criterion = model.compute_loss
    auc_meter = AUCMeter()
    scores = [ np.asarray([], dtype='float32') for i in range(2) ]
    for loader_id, loader in tqdm(enumerate((neg_batch_loader, pos_batch_loader))):
        for data in tqdm(loader):
            data = map(corpus.get, data)
            batch = None
            if not args.eval_use_content:
                batch = (batchify(data[0][0]), batchify(data[1][0]))
            else:
                batch = (map(batchify, data[0]), map(batchify, data[1]))
                new_batch = [ ]
                for x in batch:
                    for y in x:
                        new_batch.append(y)
                batch = new_batch
            labels = torch.ones(batch[0].size(1)).type(torch.LongTensor)*loader_id
            if args.cuda:
                batch = [ x.cuda() for x in batch ]
                labels = labels.cuda()
            if not args.eval_use_content:
                batch = (Variable(batch[0], volatile=True), Variable(batch[1], volatile=True))
            else:
                batch = (Variable(batch[0], volatile=True), Variable(batch[1], volatile=True), Variable(batch[2], volatile=True), Variable(batch[3], volatile=True))
            labels = Variable(labels)
            if not args.eval_use_content:
                repr_left = model(batch[0])
                repr_right = model(batch[1])
            else:
                repr_left = model(batch[0]) + model(batch[1])
                repr_right = model(batch[2]) + model(batch[3])
            output = model.compute_similarity(repr_left, repr_right)

            if model.criterion.startswith('classification'):
                assert output.size(1) == 2
                output = nn.functional.log_softmax(output)
                current_scores = -output[:,loader_id].data.cpu().squeeze().numpy()
                output = output[:,1]
            else:
                assert output.size(1) == 1
                current_scores = output.data.cpu().squeeze().numpy()
            auc_meter.add(output.data, labels.data)
            scores[loader_id] = np.append(scores[loader_id], current_scores)

    auc_score = auc_meter.value()
    auc10_score = auc_meter.value(0.1)
    auc05_score = auc_meter.value(0.05)
    auc02_score = auc_meter.value(0.02)
    auc01_score = auc_meter.value(0.01)
    if model.criterion.startswith('classification'):
        avg_score = (scores[1].mean()+scores[0].mean())*0.5
    else:
        avg_score = scores[1].mean()-scores[0].mean()
    outputManager.say("\r[{}] auc(.01): {:.3f}  auc(.02): {:.3f}  auc(.05): {:.3f}"
            "  auc(.1): {:.3f}  auc: {:.3f}"
            "  scores: {:.2f} ({:.2f} {:.2f})\n".format(
        os.path.basename(filepath).split('.')[0],
        auc01_score,
        auc02_score,
        auc05_score,
        auc10_score,
        auc_score,
        avg_score,
        scores[1].mean(),
        scores[0].mean()
    ))

    if logging:
        s = summary.scalar('auc', auc_score)
        valid_writer.add_summary(s, iter_cnt)
        s = summary.scalar('auc (fpr<0.1)', auc10_score)
        valid_writer.add_summary(s, iter_cnt)
        s = summary.scalar('auc (fpr<0.05)', auc05_score)
        valid_writer.add_summary(s, iter_cnt)
        s = summary.scalar('auc (fpr<0.02)', auc02_score)
        valid_writer.add_summary(s, iter_cnt)
        s = summary.scalar('auc (fpr<0.01)', auc01_score)
        valid_writer.add_summary(s, iter_cnt)
        valid_writer.close()

    return auc05_score

def main(args):
    model_class = get_model_class(args.model)
    model_class.add_config(argparser)
    args = argparser.parse_args()

    args.run_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    root_dir = os.path.dirname(os.path.realpath(__file__))
    # only use generated run_path if none provided by user
    if args.run_path is None:
        args.run_path = os.path.join(root_dir, args.run_dir, args.run_id)
    if not os.path.exists(args.run_path):
        os.makedirs(args.run_path)

    global outputManager
    outputManager = OutputManager(args.run_path)

    outputManager.say(args)


    #if not os.path.exists(args.run_dir):
    #    os.makedirs(args.run_dir)
    #assert os.path.isdir(args.run_dir)
    #assert not os.path.exists(args.run_path)
    #os.makedirs(args.run_path)
    outputManager.say("\nRun ID: {}\nRun Path: {}\n\n".format(
        args.run_id,
        args.run_path
    ))


    train_corpus_path = os.path.dirname(args.train) + "/corpus.tsv.gz"
    train_corpus = Corpus([ tuple([train_corpus_path, os.path.dirname(args.train)]) ])
    valid_corpus_path = os.path.dirname(args.eval) + "/corpus.tsv.gz"
    valid_corpus = Corpus([ tuple([valid_corpus_path, os.path.dirname(args.eval)]) ])
    outputManager.say("Corpus loaded.\n")

    embs = load_embedding(args.embedding) if args.embedding else None

    embedding_layer = EmbeddingLayer(args.n_d, ['<s>', '</s>'],
        embs
    )

    model = model_class(
        embedding_layer,
        args
    )

    if args.cuda:
        model.cuda()
    outputManager.say("\n{}\n\n".format(model))

    outputManager.say(model.state_dict().keys())

    needs_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(needs_grad, model.parameters()),
        lr = args.lr
    )
    outputManager.say(optimizer.state_dict())

    if args.load_model:
        outputManager.say("Loading pretrained model")
        model.load_state_dict(torch.load(args.load_model))

    else:
        outputManager.say("Training will begin from scratch")

    best_dev = 0
    iter_cnt = 0

    current_dev = evaluate(iter_cnt, args.eval+"/dev", model, valid_corpus, args)
    evaluate(iter_cnt, args.eval+"/test", model, valid_corpus, args, False)

    for epoch in range(args.max_epoch):
        iter_cnt = train(iter_cnt, model, train_corpus, args, optimizer)
        current_dev = evaluate(iter_cnt, args.eval+"/dev", model, valid_corpus, args)
        if current_dev > best_dev:
            best_dev = current_dev
            evaluate(iter_cnt, args.eval+"/test", model, valid_corpus, args, False)
        outputManager.say("\n")

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--cuda", action="store_true")
    argparser.add_argument("--run_dir",  type=str, default="/D/home/tao/mnt/ASAPPNAS/tao/test")
    argparser.add_argument("--run_path", type=str, default=None)
    argparser.add_argument("--model", type=str, required=True, help="which model class to use")
    argparser.add_argument("--embedding", "--emb", type=str, help="path of embedding")
    argparser.add_argument("--train", type=str, required=True, help="training file")
    argparser.add_argument("--eval", type=str, required=True, help="validation file")
    argparser.add_argument("--batch_size", "--batch", type=int, default=100)
    argparser.add_argument("--max_epoch", type=int, default=100)
    argparser.add_argument("--learning", type=str, default='adam')
    argparser.add_argument("--lr", type=float, default=0.0005)
    argparser.add_argument("--use_content", action="store_true")
    argparser.add_argument("--eval_use_content", action="store_true")
    argparser.add_argument("--load_model", type=str, required=False, help="load a pretrained model")
    argparser.add_argument("--save_model", type=str, required=False, help="location to save model")

    args, _  = argparser.parse_known_args()
    try:
        main(args)
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
