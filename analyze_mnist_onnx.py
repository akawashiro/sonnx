import os
os.environ["OMP_NUM_THREADS"] = "1"

import onnxruntime
import chainer
import numpy as np
import time
import onnx
import matplotlib.pyplot as plt
from onnx import shape_inference, numpy_helper
from google.protobuf.json_format import MessageToJson

MODEL_PATH = "mnist.onnx"


def dump_mnist():
    train, test = chainer.datasets.get_mnist(withlabel=True, ndim=3)
    with open("mnist_train.txt", "w") as f:
        for i, t in enumerate(train):
            for x in t[0].flatten():
                print(x, " ", end="", file=f)
            print(t[1], file=f)

    with open("mnist_test.txt", "w") as f:
        for i, t in enumerate(test):
            for x in t[0].flatten():
                print(x, " ", end="", file=f)
            print(t[1], file=f)


def dump_mnist_onnx_matrix(model_path):
    model = onnx.load(model_path)
    for number, i in enumerate(model.graph.initializer):
        t = numpy_helper.to_array(i)
        p = i.name + "_matrix.txt"
        with open(p, 'w') as f:
            if len(i.dims) == 2:
                for r in t:
                    for x in r:
                        print(str(x) + " ", end="", file=f)
                    print("", file=f)
            if len(i.dims) == 1:
                for x in t:
                    print(str(x) + " ", end="", file=f)


def jsonize_mnist_onnx(model_path):
    model = onnx.load(model_path)
    j = MessageToJson(model)
    with open(model_path + ".json", "w") as f:
        print(j, file=f)


def visualize_mnist_onnx(model_path):
    model = onnx.load(model_path)
    for number, i in enumerate(model.graph.initializer):
        t = numpy_helper.to_array(i)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(t.flatten(), bins=500)
        plt.savefig(i.name + '_figure.png')


def show_mnist_onnx(model_path):
    model = onnx.load(model_path)
    # モデル（グラフ）を構成するノードを全て出力する
    print("====== Nodes ======")
    for i, node in enumerate(model.graph.node):
        print("[Node #{}]".format(i))
        print(node)
        print(type(node))
        print(node.input)
        print(node.output)
        for a in node.attribute:
            print('a.g = ', a.g)
            print('a.graphs = ', a.graphs)
            print(a)
            print(a.doc_string)
            print(a.name)
            print(a.t)
            print(a.tensors)
            print(a.floats)
            print(a.ints)

    # モデルの入力データ一覧を出力する
    print("====== Inputs ======")
    for i, input in enumerate(model.graph.input):
        print("[Input #{}]".format(i))
        print(input)

    # モデルの出力データ一覧を出力する
    print("====== Outputs ======")
    for i, output in enumerate(model.graph.output):
        print("[Output #{}]".format(i))
        print(output)

    inferred_model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred_model)
    print('After shape inference, the shape info of Y is:\n{}'.format(
        inferred_model.graph.value_info))


def run_mnist_onnx(MODEL_PATH):
    session = onnxruntime.InferenceSession(MODEL_PATH)
    session.get_modelmeta()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    train, test = chainer.datasets.get_mnist(withlabel=True, ndim=3)

    t_start = time.time()
    correct = 0
    for t in test:
        o = session.run([output_name], {input_name: t[0]})[0]
        i = np.argmax(o[0])
        if i == t[1]:
            correct += 1
    t_end = time.time()
    print("Accuracy rate = ", float(correct) /
          float(len(test)), ", time = ", t_end - t_start)


if __name__ == '__main__':
    visualize_mnist_onnx(MODEL_PATH)
    jsonize_mnist_onnx(MODEL_PATH)
    dump_mnist_onnx_matrix(MODEL_PATH)
    run_mnist_onnx(MODEL_PATH)
    dump_mnist()
    # show_mnist_onnx(MODEL_PATH)
