# gala コンペ
https://www.kaggle.com/c/gala-images-classification/overview


## 大きく効いたこと
### pre-processing
- 普通にリサイズしたほうが精度が良い（PadIfneeded は逆効果。おそらく中央に対象があることが多いため？）
- 200x160など少し大きめにリサイズしたほうが精度がよくなった（おそらくモデルのサイズに合わせた大きさにすると良いのか？過学習かも）。しかし大きすぎるのも逆効果。あとで考えたら、CoarseDropout,cutout による穴が大きすぎたせいかもしれない。
- CoarseDropout,cutout は両方やるより cutout のみの方がよかった(確率の問題か。それぞれの適用確率が0.5だったので、適用確率が0.75 になって高すぎたっぽい?)

### モデル
- 4つ全て学習する（詳細は後述。misc を除いて閾値で除外とかでもあまり効果なし）
- fine tuning をする（小さめのモデルである efficient net b1 ns が一番良かった）


## 試したけどうまくいかなかったこと

### モデル: misc を分けて学習
思想：misc はノイズ(雑多な画像が含まれている)ので分けて学習する

- 2種クラス分類器 misc
- 3種クラス分類器 clean

の2つを組み合わせる

参考：https://www.slideshare.net/kensukemitsuzawa/ss-69696373

#### 2種分類器 misc
misc or その他(clean) の、2種クラス分類を行う
→ galaに分類されたものは、アーキテクチャ clean でFood, Attire, Decorationandsignage のどれかを分類する

#### 3種分類器 clean
misc 以外の3種(clean)のみを学習に用いる

### clean のみをつかって予測
miscは出力の確率が大きくないときに予測
（train の misc に対して、どの閾値を設定するのが良いかを試す.また、val の評価に閾値ごとの性能評価ができると良い）


### 上述のモデルを色々アンサンブル
LightGBM でアンサンブル。しかしあまりうまくいかなかった
