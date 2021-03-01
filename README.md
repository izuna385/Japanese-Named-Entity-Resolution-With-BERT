# BERTを用いた名寄せ解消
* https://github.com/yagays/nayose-wikipedia-ja　のデータセットを用いた名寄せ解消

# モデル
* [InferSent](https://github.com/facebookresearch/InferSent) を一部参考に以下のモデルを組んだ。

![model](./img/model.png)

# 結果
* dev acc ~ 50% なので要改良。
* 例えば [この論文](https://www.aclweb.org/anthology/2020.emnlp-main.363/) で使用している特徴量を組み込むことが考えられる。

