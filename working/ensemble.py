'''
同じように stratifyKFold で分割
configs を元に、all, clean, misc のモデルを推論
Lightgbm などを用いてアンサンブル
'''
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  log_loss

# LightGBM parameters
params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': {'multi_logloss'},
        'num_class': 4,
        'learning_rate': 0.01,
        'max_depth': 4,
        'num_leaves':3,
        'lambda_l2' : 0.3,
        'num_iteration': 2000,
        "min_data_in_leaf":1,
        'verbose': 0
}

CFG = {
    "fold_num": 5,
    "seed": 719,
}

# data path
oof_path = [
    "./output/default_tf_efficientnet_b1_ns_oof.csv",
    "./output/clean_tf_efficientnet_b1_ns_oof.csv",
    "./output/clean_resize_tf_efficientnet_b1_ns_oof.csv",
    "./output/misc_tf_efficientnet_b1_ns_oof.csv",
    "./output/misc_resize_tf_efficientnet_b1_ns_oof.csv",
    "./output/all_tf_efficientnet_b2_ns_oof.csv",
    "./output/pad_tf_efficientnet_b1_ns_oof.csv",
    "./output/resize_tf_efficientnet_b1_ns_oof.csv",
]
test_path = [
    "./output/default_tf_efficientnet_b1_ns_test.csv",
    "./output/clean_tf_efficientnet_b1_ns_test.csv",
    "./output/clean_resize_tf_efficientnet_b1_ns_test.csv",
    "./output/misc_tf_efficientnet_b1_ns_test.csv",
    "./output/misc_resize_tf_efficientnet_b1_ns_test.csv",
    "./output/all_tf_efficientnet_b2_ns_test.csv",
    "./output/pad_tf_efficientnet_b1_ns_test.csv",
    "./output/resize_tf_efficientnet_b1_ns_test.csv",
]

class LightGBM():
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        # データセットを生成する
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

        # 上記のパラメータでモデルを学習する
        model = lgb.train(
            params, lgb_train,
            # モデルの評価用データを渡す
            valid_sets=lgb_eval,
            # 50 ラウンド経過しても性能が向上しないときは学習を打ち切る
            early_stopping_rounds=50,
        )

        # valid を予測する
        y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        # テストデータを予測する
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        return y_pred, y_valid_pred, model

def load_df(oof_path, test_path):
    oof_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for path in oof_path:
        one_df = pd.read_csv(path)
        oof_df = pd.concat([oof_df, one_df], axis=1)
    for path in test_path:
        one_df = pd.read_csv(path)
        test_df = pd.concat([test_df, one_df], axis=1)
    return oof_df, test_df

def load_train_label(path="../input/gala-images-classification/dataset/train.csv"):
    train_with_misc = pd.read_csv(path)
    train_with_misc["is_misc"] = (train_with_misc["Class"]=="misc")*1
    label_dic ={"Attire":0, "Food":1, "Decorationandsignage":2,"misc":3}
    train_with_misc["label"]=train_with_misc["Class"].map(label_dic)
    return train_with_misc["label"]

def main():
    oof_df, test_df = load_df(oof_path, test_path)
    oof_label = load_train_label()

    y_preds = []
    scores_loss = []
    scores_acc = []
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(oof_df.shape[0]), oof_label.values)
    for fold, (trn_idx, val_idx) in enumerate(folds):
        X_train, X_valid = (oof_df.iloc[trn_idx, :], oof_df.iloc[val_idx, :])
        y_train, y_valid = (oof_label.iloc[trn_idx], oof_label.iloc[val_idx])

        model = LightGBM()

        # 学習と推論
        y_pred, y_valid_pred, m = model.train_and_predict(X_train, X_valid, y_train, y_valid, test_df, params)

        # 結果の保存
        y_preds.append(y_pred)

        # スコア
        loss = log_loss(y_valid, y_valid_pred)
        scores_loss.append(loss)
        acc = (y_valid == np.argmax(y_valid_pred, axis=1)).mean()
        scores_acc.append(acc)
        print(f"\t log loss: {loss}")
        print(f"\t acc: {acc}")

    loss = sum(scores_loss) / len(scores_loss)
    print('===CV scores loss===')
    print(scores_loss)
    print(loss)
    acc = sum(scores_acc) / len(scores_acc)
    print('===CV scores acc===')
    print(scores_acc)
    print(acc)

    tst_preds = np.mean(y_preds, axis=0)

    test = pd.DataFrame()
    test['Image'] = list(os.listdir('../input/gala-images-classification/dataset/Test_Images'))
    test['Class'] = np.argmax(tst_preds, axis=1)
    label_dic = {0:"Attire", 1:"Food", 2:"Decorationandsignage",3:"misc"}
    test["Class"] = test["Class"].map(label_dic)
    print(test.value_counts("Class"))
    test.to_csv(f'output/submission_ensemble.csv', index=False)

if __name__ == '__main__':
    main()
