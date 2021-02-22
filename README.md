# gala コンペ

## アーキテクチャ
思想：miscはノイズなので分けて学習する

- 2種クラス分類器 misc
- 3種クラス分類器 clean

の2つを組み合わせる

参考：https://www.slideshare.net/kensukemitsuzawa/ss-69696373

### 2種分類器 misc
gala or misc の、2種クラス分類を行う
→galaに分類されたものは、アーキテクチャ clean でFood, Attire, Decorationandsignage のどれかを分類する

### 3種分類器 clean
普通の3種を学習に用いて、miscは出力の確率が大きくないときに予測
（train の misc に対して、どの閾値を設定するのが良いかを試す.また、val の評価に閾値ごとの性能評価ができると良い）


## イメージのリサイズ・オーグメンテーション
イメージのサイズの揃え方はいくつかバリエーションを作りたい
・最小画像にリサイズ
・最大画像にリサイズ
・最大画像にPadIfNeeded
