import onnx
import matplotlib.pyplot as plt
from onnx import helper, shape_inference
from onnx import TensorProto
from onnx import numpy_helper

MODEL_PATH = "mnist.onnx"


def dump_onnx_matrix(model_path):
    model = onnx.load(model_path)
    print(type(model))
    for number, i in enumerate(model.graph.initializer):
        print(type(i))
        print(i.doc_string)
        print(i.name)
        print(i.dims)
        t = numpy_helper.to_array(i)
        p = i.name + "_matrix.txt"
        with open(p, 'w') as f:
            if len(i.dims) == 2:
                for r in t:
                    for x in r:
                        print(str(x) + " ", end="", file=f)
                    print("", file=f)
            if len(i.dims) == 1:
                print(i.dims)
                for x in t:
                    print(str(x) + " ", end="", file=f)
        print(len(t))


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


if __name__ == "__main__":
    dump_onnx_matrix(MODEL_PATH)
