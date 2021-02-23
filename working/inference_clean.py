# misc を除いて、clean なデータのみで学習する
import argparse
import json
import os
import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  log_loss

# 引数で config の設定を行う
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs_infer/clean.json')
options = parser.parse_args()
CFG = json.load(open(options.config))

# logger の設定
from logging import getLogger, StreamHandler,FileHandler, Formatter, DEBUG, INFO
logger = getLogger("logger")    #logger名loggerを取得
logger.setLevel(DEBUG)  #loggerとしてはDEBUGで
#handler1を作成
handler_stream = StreamHandler()
handler_stream.setLevel(DEBUG)
handler_stream.setFormatter(Formatter("%(asctime)s: %(message)s"))
#handler2を作成
config_filename = os.path.splitext(os.path.basename(options.config))[0]
handler_file = FileHandler(filename=f'./logs/log_inference_clean_{config_filename}_{CFG["model_arch"]}.log')
handler_file.setLevel(DEBUG)
handler_file.setFormatter(Formatter("%(asctime)s: %(message)s"))
#loggerに2つのハンドラを設定
logger.addHandler(handler_stream)
logger.addHandler(handler_file)

from model.transform import get_train_transforms, get_valid_transforms, get_inference_transforms
from model.dataset import GalaDataset
from model.model import GalaImgClassifier
from model.epoch_api import train_one_epoch, valid_one_epoch, inference_one_epoch
from model.utils import seed_everything


test = pd.DataFrame()
test['Image'] = list(os.listdir('../input/gala-images-classification/dataset/Test_Images'))


def load_clean_train_df(path):
    "infer_clean 用。ひとまず全て読みだす(miscも含む）"
    train_with_misc = pd.read_csv(path)
    train_with_misc["is_misc"] = (train_with_misc["Class"]=="misc")*1
    label_dic = {"Attire":0, "Food":1, "Decorationandsignage":2,"misc":3}
    train_with_misc["label"]=train_with_misc["Class"].map(label_dic)
    return train_with_misc

def infer_clean():
    logger.debug("pred clean start")
    train = load_clean_train_df("../input/gala-images-classification/dataset/train.csv")
    seed_everything(CFG['seed'])

    folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), train.label.values)

    tst_preds = []
    val_loss = []
    val_acc = []
    for fold, (trn_idx, val_idx) in enumerate(folds):
        logger.debug('Inference fold {} started'.format(fold))

        input_shape=(CFG["img_size_h"], CFG["img_size_w"])
        valid_ = train.loc[val_idx,:].reset_index(drop=True)
        valid_ds = GalaDataset(valid_, '../input/gala-images-classification/dataset/Train_Images', transforms=get_inference_transforms(input_shape), shape=input_shape, output_label=False)

        # misc でないと判断したものを推論する
        test_ds = GalaDataset(test, '../input/gala-images-classification/dataset/Test_Images', transforms=get_inference_transforms(input_shape), shape=input_shape, output_label=False)

        val_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        tst_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        device = torch.device(CFG['device'])
        #model = GalaImgClassifier(CFG['model_arch'], train.label.nunique()).to(device)
        model = GalaImgClassifier(CFG['model_arch'], train.label.nunique()-1).to(device) # misc を除いた

        val_preds = []

        #for epoch in range(CFG['epochs']-3):
        for i, epoch in enumerate(CFG['used_epochs_clean']):
            model.load_state_dict(torch.load('save/clean_{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch)))

            with torch.no_grad():
                for _ in range(CFG['tta']):
                    val_preds += [CFG['weights_clean'][i]/sum(CFG['weights_clean'])*inference_one_epoch(model, val_loader, device)]
                    tst_preds += [CFG['weights_clean'][i]/sum(CFG['weights_clean'])*inference_one_epoch(model, tst_loader, device)]

        val_preds = np.mean(val_preds, axis=0)
        # misc を除いたときの validation loss をみる
        indx = valid_["Class"]!="misc"
        val_loss.append(log_loss(valid_.label.values[indx], val_preds[indx]))
        val_acc.append((valid_.label.values[indx]==np.argmax(val_preds[indx], axis=1)).mean())
        # 閾値ごとの正答率をみる
        for p in CFG["prob_thres"]:
            label_preds = np.argmax(val_preds, axis=1)
            label_preds[val_preds.max(axis=1)<p] = 3
            logger.debug('fold {} (p={}) validation accuracy = {:.5f}'.format(fold,p,(valid_.label.values==label_preds).mean()))
    logger.debug('no misc validation loss = {:.5f}'.format( np.mean(val_loss)))
    logger.debug('no misc validation accuracy = {:.5f}'.format( np.mean(val_acc)))
    tst_preds = np.mean(tst_preds, axis=0)
    del model
    torch.cuda.empty_cache()
    return tst_preds

if __name__ == '__main__':
    logger.debug(CFG)
    tst_preds = infer_clean()

    # 予測結果を保存
    for p in CFG["prob_thres"]:
        logger.debug(f"p: {p}")
        test['Class'] = np.argmax(tst_preds, axis=1)
        test.loc[tst_preds.max(axis=1)<p, 'Class'] = 3
        label_dic = {0:"Attire", 1:"Food", 2:"Decorationandsignage",3:"misc"}
        test["Class"]=test["Class"].map(label_dic)
        logger.debug(test.value_counts("Class"))
        test.to_csv(f'output/submission_clean_{p}_{config_filename}_{CFG["model_arch"]}.csv', index=False)
