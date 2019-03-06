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
実際、学習後のモデルで赤丸で囲った部分の重みの分布は
<img src="https://raw.githubusercontent.com/akawashiro/sonnx/master/designate_layer.png" width=400px>

このようになっています。
<img src="https://raw.githubusercontent.com/akawashiro/sonnx/master/140406443536680_figure.png" width=400px>
一見してわかるように、重みが0の部分が非常に多いです。まず、学習済みモデルから重みが0の枝を削除しても推論結果に影響しないはずです。また、グラフが左右対称になっているので、絶対値の小さい順に削除していけば、各パーセプトロンへの入力はそれほど変化しない気がします。

## 手法
やることは非常に単純です。ニューラルネットワーク中の各層間枝の重みの統計を取り、重み上位何%かを残して残りは削除するだけです。

しかし、巷のディープラーニングフレームワークはこのような処理を行うことができません。そこで、何らかの方法で学習済みのモデルから枝の重みのデータを取り出し、辺をカットし、さらに加工後のモデルのデータを使って推論できるようにする必要があります。

幸い今はONNXという良いものがあります。ONNXとは学習済みモデルのデータを出力する形式の一つで、多くのフレームワークが対応しています。

今回はChainerで書いたモデルからONNXデータを出力し、そのデータを加工することにしました。また加工後のデータはC++で書いた俺俺ONNXランタイムに実行してもらうことにしました。

纏めると、
1. Chainerでニューラルネットワークを書いて学習する
2. 学習済みのニューラルネットワークからONNXデータを出力する
3. ONNXデータを俺俺ONNXランタイムに読み込んで加工、実行する

となります。一つづつ何をやるのかを説明します。

#### 1. Chainerでニューラルネットワークを書いて学習する
#### 2. 学習済みのニューラルネットワークからONNXデータを出力する
1と2は簡単です。[onnx-chainer](https://github.com/chainer/onnx-chainer)を使えばすぐにできます。[このソースコード](https://github.com/akawashiro/sonnx/blob/master/learn_mnist.py)を`python3 learn_mnist.py`で実行すると`mnist.onnx`というファイルができます。

#### 3. ONNXデータを俺俺ONNXランタイムに読み込んで加工、実行する
ここが大変でした。ONNXモデルの出力は多くの人が試しているのですが、出力したONNXモデルをチューニングしようとする人はほとんどいないようです。
##### 3.1 ONNXデータを解析する
とりあえず[netron](https://github.com/lutzroeder/netron)というONNXの可視化ツールで`mnist.onnx`を可視化してみました。
<img src="https://raw.githubusercontent.com/akawashiro/sonnx/master/mnist.png" width=250px>
`Gemm`はGeneral matrix multiplyの略です。各Gemmノードは行列`B`とベクトル`C`を持ち、ベクトル`x`を入力として`Bx+C`を出力します。`Relu`は[活性化関数](https://ja.wikipedia.org/wiki/%E6%B4%BB%E6%80%A7%E5%8C%96%E9%96%A2%E6%95%B0)です。

Reluは`max(0,x)`で定義されているので、各Gemmノードの行列`B`とベクトル`C`の情報を抽出できれば良さそうです。

## 結果
<img src="https://raw.githubusercontent.com/akawashiro/sonnx/master/sonnx_result.png" width=500px>
## 考察
## おまけ
- numpyには勝てなかったよ...