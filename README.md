# gala コンペ

## 効いたこと
pre-processing
・普通にリサイズしたほうが精度が良い（PadIfneeded は逆効果。おそらく中央に対象があることが多いため？）
・少し大きめにリサイズしたほうが精度がよくなった（おそらくモデルのサイズに合わせた大きさにすると良いのか？）
・CoarseDropout,cutout は両方やるより cutout のみの方がよかった(確率の問題。適用確率が0.75 になって高すぎたっぽい)

モデル
・4つ全て学習する（misc を除いて閾値で除外とかでもあまり効果なし）
・fine tuning をする（efficient net b1 ns が一番良かった）


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
