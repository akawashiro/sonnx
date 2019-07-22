## MNISTを可能な限り高速に分類する

## 概要

- MNISTの分類をする学習済みモデルを軽量化し、
- 更にSIMD命令を使った高速化を行うことで、
- `onnxruntime`より高速なMNISTの分類が可能になりました

## 前回までのあらすじ

この記事は[前回](http://a-kawashiro.hatenablog.com/entry/2019/03/07/201304)の続きです。
前回はMNISTを分類する学習済みニューラルネットワークから不要な枝を削除し、軽量化した学習済みモデルを走らせる専用のランタイム[sonnx](https://github.com/akawashiro/sonnx)を作ってMNIST分類の高速化を試みました。
今回はSIMD命令とマルチスレッド化による最適化でMNIST分類速度の限界に挑みます。

今回の目標タイムは既存のONNXランタイム[onnxruntime](https://github.com/microsoft/onnxruntime)です。
この記事の各実行時間の計測は10回行い、その平均と分散を求めました。
onnxruntimeと最適化無しのsonnxの実行時間は、OS: Ubuntu19.04、CPU: Core i7 8th Gen、 メモリ: 16GiB、GPU無しの環境下で以下のようになりました。[^1]

| 手法                          | 実行時間                     |
|-------------------------------|------------------------------|
| onnxruntime(シングルスレッド) | 1.259秒 (標準偏差 0.1148秒)  |
| onnxruntime(マルチスレッド)   | 0.505秒 (標準偏差 0.04249秒) |
| sonnx(最適化無し)             | 6.968秒 (標準偏差 0.08912秒) |

[^1]: [前回](http://a-kawashiro.hatenablog.com/entry/2019/03/07/201304)からOSを入れ替えたので数値が違います。


機械学習の推論過程、特にMNISTを分類する場合、においてボトルネックになるのは重み行列を入力ベクトルに乗算する処理です。
[前回](http://a-kawashiro.hatenablog.com/entry/2019/03/07/201304)は重み行列の中で絶対値の小さい要素を無視することで計算の効率化を図り、行列の要素の80%を無視して約5倍の高速化に成功しました。
しかし、80%を削減してもなお乗算はボトルネックでした。

具体的には[sonnx.cpp](https://github.com/akawashiro/sonnx/blob/master/sonnx.cpp)のこの部分がボトルネックになります。
```c
for(int i=0;i<n;i++){
        ret[B_row[i]] += B_scale[i] * x[B_column[i]];
}
```

## SIMDによる高速化

まず、sonnxをSIMDで高速化してみます。
Core i7 8th GenではAVX2命令セットが使えるので256bitの演算を一度に行うことができ、
今回は32bit浮動小数点数で計算しているので最大8倍の高速化が見込めます。

しかし、元のコードはメモリへの間接参照を２つ(`ret[B_row[i]]`と`x[B_column[i]]`)含んでおりそのままではSIMD化するのが困難です。
まず`ret[B_row[i]]`への書き込みは同じ`B_row[i]`の値を持つものをまとめて計算し、メモリへの書き込みを一度に行います。
`x[B_column[i]]`からの読み出しはAVX2のgather命令を使ってSIMD化します。
完成したコードが[これ](https://github.com/akawashiro/sonnx/blob/avx2/sonnx.cpp)です。

```c
float r = 0;
__m256 vr = _mm256_setzero_ps();
for(cur;cur<n1;cur+=8){
    __m256i vc = _mm256_loadu_si256((__m256i*)(&B_column_p[cur]));
    __m256 vx = _mm256_i32gather_ps(x_p, vc, 4);
    __m256 vs = _mm256_load_ps(&B_scale_p[cur]);
    vr = _mm256_fmadd_ps(vs, vx, vr);
}
for(cur;cur<n;cur++){
    r += B_scale_p[cur] * x_p[B_column_p[cur]];
}
__attribute__((aligned(32))) float t[8] = {0};
_mm256_store_ps(t, vr);
ret[B_row_p[cur-1]] += r + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
```

実行時間はこんな感じになりました。

| 手法                          | 実行時間                    |
|-------------------------------|-----------------------------|
| sonnx(SIMD)                   | 1.121秒(標準偏差 0.02271秒) |
| onnxruntime(シングルスレッド) | 1.259秒(標準偏差 0.1148秒)  |

ウェルチのt検定を用いて検定すると[^2]有意水準5%で帰無仮説が棄却され、確かにonnxruntimeより高速に推論できています。
onnxruntime(シングルスレッド)と大きな差がつかないのはgather命令が遅いからでしょうか。

[^2]: ちょっとここ正しく議論できているか自信がない


## マルチスレッドによる高速化

(SIMDなしの)マルチスレッド化はどうでしょうか。
オリジナルのコードはメモリへの書き込みが一箇所(`ret[B_row[i]]`への書き込み)しかないので、この部分だけ複数スレッドで同時に書き込まないようにすればデータレースは起こりません。
スレッド数は手元で最適なものを探索した結果4にしています。
ソースコードは[ここ](https://github.com/akawashiro/sonnx/blob/multithread/sonnx.cpp)にあります。

```c
void CompressedGemm::calc_partially(const int index, const vector<float> &x){
    int m = B_nrows_threads[index].size();
    int cur = 0;
    for(int i=0;i<m;i++){
        int n = B_nrows_threads[index][i];
        float r = 0;
        for(int j=cur;j<cur+n;j++){
            r += B_scale_threads[index][j] * x[B_column_threads[index][j]];
        }
        ret[B_row_threads[index][cur]] += r;
        cur += n;
    }
}

vector<float> CompressedGemm::calc(const vector<float> &x){
    ret = C;
    vector<thread> ths;
    
    for(int i=0;i<n_thread;i++){
        ths.push_back(thread(&CompressedGemm::calc_partially, this, i, x));
    }
    for(int i=0;i<n_thread;i++){
        ths[i].join();
    }
    return ret;
}
```

結果はこんな感じです。

| 手法                        | 実行時間                     |
|-----------------------------|------------------------------|
| sonnx(マルチスレッド)       | 1.995秒(標準偏差 0.03405秒)  |
| onnxruntime(マルチスレッド) | 0.5048秒(標準偏差 0.04249秒) |

完敗です。

## SIMDとマルチスレッドの併用

では最後にSIMDとマルチスレッディングを併用してみます。
ソースコードは[ここ](https://github.com/akawashiro/sonnx/blob/multithread+AVX2/sonnx.cpp)にあります。

| 手法                        | 実行時間                     |
|-----------------------------|------------------------------|
| sonnx(SIMD+マルチスレッド)  | 1.358秒(標準偏差 0.05762秒)  |
| onnxruntime(マルチスレッド) | 0.5048秒(標準偏差 0.04249秒) |

あまり効果がありませんね...複数コアからの結果をまとめるのに時間がかかっているのかもしれません。

## まとめ
影響力の小さい要素を削除して学習済みのモデルを軽量化した上でSIMDとマルチスレッド化を用いて推論の高速化を試みました。
マルチスレッド環境ではonnxruntimeに勝てませんでしたが、シングルスレッドではonnxruntimeより高速に推論できました。

## 参考
- [実行時間の測定データ](https://github.com/akawashiro/sonnx/blob/multithread%2BAVX2/optimization-result.ods)
