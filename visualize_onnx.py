import onnx
from onnx import helper, shape_inference
from onnx import TensorProto
from onnx import numpy_helper

model_path = "mnist.onnx"


def main():
    # ONNX形式のモデルを読み込む
    model = onnx.load(model_path)
    print(type(model))
    print(model.doc_string)
    print(model.domain)
    print(model.ir_version)
    print(model.model_version)
    print(model.producer_version)
    for i in model.graph.initializer:
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
        print(t)

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


if __name__ == "__main__":
    main()
