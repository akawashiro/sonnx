import onnx
import matplotlib.pyplot as plt
from onnx import helper, shape_inference
from onnx import TensorProto
from onnx import numpy_helper

MODEL_PATH = "mnist.onnx"


def visualize_mnist_onnx_model(model_path):
    # ONNX形式のモデルを読み込む
    model = onnx.load(model_path)
    print(type(model))
    print(model.doc_string)
    print(model.domain)
    print(model.ir_version)
    print(model.model_version)
    print(model.producer_version)
    for number, i in enumerate(model.graph.initializer):
        # print(i.data_type)
        print(i.dims)
        # print(i.float_data)
        # print(i.double_data)
        # print(i.int32_data)
        # print(i.int64_data)
        # print(i.raw_data)
        t = numpy_helper.to_array(i)
        # new_tensor = onnx.TensorProto()
        print(type(i))
        print(type(t))
        # print(t.flatten())
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(t.flatten(), bins=500)
        plt.savefig(str(number) + '-' + str(len(t.flatten())) + '-figure.png')

        print(i.data_type)

        # for x in t.flatten():
        #     print(x)


def compress_mnist_onnx_model(model_path):
    model = onnx.load(model_path)
    print(type(model.graph.initializer))
    for index, init in enumerate(model.graph.initializer):
        # print(init.dims)
        if init.dims == [1000, 1000]:
            t = numpy_helper.to_array(init)
            print(type(t))
            s = t.flatten()
            s = sorted(s, key=lambda x: abs(x))
            th = s[len(s)//5*4]
            print(th)
            for i in range(1000):
                for j in range(1000):
                    if abs(t[i][j]) < th:
                        t[i][j] = 0
            init = numpy_helper.from_array(t)
            model.graph.initializer[index] = init


if __name__ == '__main__':
    compress_mnist_onnx_model(MODEL_PATH)
    # visualize_mnist_onnx_model(MODEL_PATH)

    # onnx.save(model, 'rewrite-mnist.onnx')

    # モデル（グラフ）を構成するノードを全て出力する
    # print("====== Nodes ======")
    # for i, node in enumerate(model.graph.node):
    #     print("[Node #{}]".format(i))
    #     print(node)
    #     print(type(node))
    #     print(node.input)
    #     print(node.output)
    #     for a in node.attribute:
    #         print('a.g = ', a.g)
    #         print('a.graphs = ', a.graphs)
    #     print(a)
    #     print(a.doc_string)
    #     print(a.name)
    #     print(a.t)
    #     print(a.tensors)
    #     print(a.floats)
    #     print(a.ints)

    # # モデルの入力データ一覧を出力する
    # print("====== Inputs ======")
    # for i, input in enumerate(model.graph.input):
    #     print("[Input #{}]".format(i))
    #     print(input)
    #
    # # モデルの出力データ一覧を出力する
    # print("====== Outputs ======")
    # for i, output in enumerate(model.graph.output):
    #     print("[Output #{}]".format(i))
    #     print(output)
    #
    # inferred_model = shape_inference.infer_shapes(model)
    # onnx.checker.check_model(inferred_model)
    # print('After shape inference, the shape info of Y is:\n{}'.format(inferred_model.graph.value_info))
