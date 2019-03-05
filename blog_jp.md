# ONNXファイルから不要な枝を削ってMNISTの推論を高速化してみる
春休みの自由研究なので先行研究とかちゃんと調べてないです。許して...
## 概要
- ニューラルネットワークから要らなそうな枝を削除してみても案外動いた
- ナイーブ実装の5倍速まで達成した
- numpyには勝てなかった
## 背景
機械学習の学習済みモデルを小さなデバイスで動かす、というのが最近流行っているそうです。機械学習では、学習には大きな計算コストがかかりますが、推論はそれほど大きな計算コストがかかりません。このため、学習だけを別のコンピュータで行っておいて、実際の推論は小さなデバイスで行うということが可能です。

ただし、推論だけでもそれなりに計算資源が必要です。そこで、学習済みのモデルの高速化が重要になります。Raspberry Piに搭載されているGPUを使う[Idein](https://blog.idein.jp/post/172704317595/idein%E3%81%AE%E6%8A%80%E8%A1%93%E3%82%84%E4%BA%8B%E6%A5%AD%E3%81%AE%E7%B4%B9%E4%BB%8B)とか有名です。

僕も学習済みモデルを高速化できそうな方法を思いついたので実験してみました。
## アイデア
今回はMNISTを分類する学習済みモデルを高速化します。今回使用するモデルは次の図のようなものです。画像は28*28(=784)pxなので入力は784個、出力は各数字の確率なので10個あり、中間層が2つ挟まっています。各層間は全結合しており、活性化関数としてReluを使います。

<img src="https://raw.githubusercontent.com/akawashiro/sonnx/master/original_NN.png" width=300px>

このモデルを教師データを使って学習すると、枝の重みが変わってこんな感じになります(イメージです)。  <br/>
<img src="https://raw.githubusercontent.com/akawashiro/sonnx/master/learned_NN.png" width=300px>

僕のアイデアは学習後のネットワークから重みの小さい枝を取り去ってもちゃんと動くんじゃないか、というものです。重みの小さい枝を取り去るとこんな感じになります。
<img src="https://raw.githubusercontent.com/akawashiro/sonnx/master/compressed_NN.png" width=300px>
枝の本数が少なくなれば、必要な計算資源も少なくなり、高速化できそうな気がします。

## アイデアの裏付け
<img src="https://raw.githubusercontent.com/akawashiro/sonnx/master/designate_layer.png" width=400px>
実際、２つの中間層の間の枝の重みの分布はこのようになっています。
<img src="https://raw.githubusercontent.com/akawashiro/sonnx/master/140406443536680_figure.png" width=400px>
重みが0の部分が非常に多く、削除してしまっても大丈夫そうに見えます。
## 手法
やることは非常に単純です。
ニューラルネットワーク中の枝の重みの統計を取り、重み上位何%かを残して残りは削除するだけです。

しかし、巷のディープラーニングフレームワークはおそらくこのような処理を行うことができません。
そこで、何らかの方法で学習済みのニューラルネットワークから枝の重みのデータを取り出し、辺をカットし、さらに加工後のニューラルネットワークのデータを実行できるようにする必要があります。

幸い今はONNXといういいものがあります。
ONNXとは学習済みニューラルネットワークのデータを出力する形式の一つで多くのフレームワークが対応しています。
今回はChainerで書いたモデルからONNXデータを出力し、そのデータを加工することにしました。
また加工後のデータは俺俺ONNXランタイムに実行してもらうことにしました。

つまり
1. Chainerでニューラルネットワークを書いて学習する
2. 学習済みのニューラルネットワークからONNXデータを出力する
3. ONNXデータを俺俺ONNXランタイムに読み込んで加工、実行する

#### 1. Chainerでニューラルネットワークを書いて学習する
#### 2. 学習済みのニューラルネットワークからONNXデータを出力する
#### 3. ONNXデータを俺俺ONNXランタイムに読み込んで加工、実行する
<img src="https://raw.githubusercontent.com/akawashiro/sonnx/master/mnist.png" width=250px>

という流れになります。
## 結果
<img src="https://raw.githubusercontent.com/akawashiro/sonnx/master/sonnx_result.png" width=500px>
## 考察
## おまけ
- numpyには勝てなかったよ...