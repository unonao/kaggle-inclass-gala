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
parser.add_argument('--config', default='./configs/default.json')
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
handler_file = FileHandler(filename=f'./logs/inference_{config_filename}_{CFG["model_arch"]}.log')
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

def load_train_df(path):
    train_with_misc = pd.read_csv(path)
    train_with_misc["is_misc"] = (train_with_misc["Class"]=="misc")*1
    label_dic ={"Attire":0, "Food":1, "Decorationandsignage":2,"misc":3}
    train_with_misc["label"]=train_with_misc["Class"].map(label_dic)
    return train_with_misc

def infer():
    logger.debug("pred start")
    train = load_train_df("../input/gala-images-classification/dataset/train.csv")
    seed_everything(CFG['seed'])

    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), train.label.values)


    tst_preds = []
    val_loss = []
    val_acc = []

    # 行数を揃えた空のデータフレームを作成
    cols = ["Attire", "Food", "Decorationandsignage","misc"]
    oof_df = pd.DataFrame(index=[i for i in range(train.shape[0])],columns=cols)
    y_preds_df = pd.DataFrame(index=[i for i in range(test.shape[0])], columns=cols)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        """
        """
        if fold > 0:
            break
        logger.debug(' fold {} started'.format(fold))
        input_shape=(CFG["img_size_h"], CFG["img_size_w"])

        valid_ = train.loc[val_idx,:].reset_index(drop=True)
        valid_ds = GalaDataset(valid_, '../input/gala-images-classification/dataset/Train_Images', transforms=get_valid_transforms(input_shape,CFG["transform_way"]), shape = input_shape, output_label=False)

        test_ds = GalaDataset(test, '../input/gala-images-classification/dataset/Test_Images', transforms=get_valid_transforms(input_shape,CFG["transform_way"]),shape=input_shape, output_label=False)


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
        model = GalaImgClassifier(CFG['model_arch'], train.label.nunique()).to(device)

        val_preds = []

        #for epoch in range(CFG['epochs']-3):
        for i, epoch in enumerate(CFG['used_epochs']):
            model.load_state_dict(torch.load(f'save/all_{config_filename}_{CFG["model_arch"]}_fold_{fold}_{epoch}'))

            with torch.no_grad():
                val_preds += [CFG['weights'][i]/sum(CFG['weights'])*inference_one_epoch(model, val_loader, device)]
                tst_preds += [CFG['weights'][i]/sum(CFG['weights'])*inference_one_epoch(model, tst_loader, device)]

        val_preds = np.mean(val_preds, axis=0)
        val_loss.append(log_loss(valid_.label.values, val_preds))
        val_acc.append((valid_.label.values == np.argmax(val_preds, axis=1)).mean())
        oof_df.loc[val_idx, cols] = val_preds

    logger.debug('validation loss = {:.5f}'.format(np.mean(val_loss)))
    logger.debug('validation accuracy = {:.5f}'.format(np.mean(val_acc)))
    tst_preds = np.mean(tst_preds, axis=0)
    y_preds_df.loc[:, cols] = tst_preds #.reshape(len(tst_preds), -1)

    # 予測値を保存
    oof_df.to_csv(f'output/{config_filename}_{CFG["model_arch"]}_oof.csv', index=False)
    y_preds_df.to_csv(f'output/{config_filename}_{CFG["model_arch"]}_test.csv', index=False)

    del model
    torch.cuda.empty_cache()
    return np.argmax(tst_preds, axis=1)


if __name__ == '__main__':
    logger.debug(CFG)
    tst_preds_label_all = infer()
    print(tst_preds_label_all.shape)
    # 予測結果を保存
    test['Class'] = tst_preds_label_all
    label_dic = {0:"Attire", 1:"Food", 2:"Decorationandsignage",3:"misc"}
    test["Class"] = test["Class"].map(label_dic)
    logger.debug(test.value_counts("Class"))
    test.to_csv(f'output/submission_{config_filename}_{CFG["model_arch"]}.csv', index=False)
