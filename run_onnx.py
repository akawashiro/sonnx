import onnxruntime
import chainer
import numpy as np

if __name__ == '__main__':
    session = onnxruntime.InferenceSession("mnist.onnx")
    session.get_modelmeta()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("input = ")
    for i in session.get_inputs():
        print(i.name)
    print("output = ")
    for o in session.get_outputs():
        print(o.name)

    train, test = chainer.datasets.get_mnist(withlabel=True, ndim=3)
    print(type(train))
    print(type(train[0][0]))
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

    # 0番目のデータセットはtrain[0][0]とtrain[0][1]で構成されている
    # train[0][0]が入力、train[0][1]が正解データ
    print(train[0][0].shape)
    print(train[0][1])

    # print(train[0][0])
    output = session.run([output_name], {
                         input_name: train[0][0]})[0]
    print(output)

    print(len(train))
    print(len(test))

    correct = 0
    for t in test:
        o = session.run([output_name], {input_name: t[0]})[0]
        i = np.argmax(o[0])
        if i == t[1]:
            correct += 1
    print("Accuracy rate = ", float(correct) / float(len(test)))
