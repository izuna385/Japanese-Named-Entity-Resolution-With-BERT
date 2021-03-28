# BERTを用いた名寄せ解消
* https://github.com/yagays/nayose-wikipedia-ja　
* 上記データセットを用いた名寄せ解消

# モデル
* [InferSent](https://github.com/facebookresearch/InferSent) を一部参考に以下のモデルを組んだ。

![model](./img/0303_model.png)

# 結果
* dev acc. ~%85 ,test acc. ~ 85%. 要改良。

* 例えば [この論文](https://www.aclweb.org/anthology/2020.emnlp-main.363/) で使用している特徴量を組み込むことが考えられる。

# 実験方法
```
$ conda create -n allennlp python=3.7
$ conda activate allennlp
$ pip install -r requirements.txt
$ sh download.sh
$ python3 main.py -num_epochs 3
```