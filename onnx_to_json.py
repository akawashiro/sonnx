import onnx
from google.protobuf.json_format import MessageToJson

model = onnx.load('./mnist.onnx')

j = MessageToJson(model)
print(j)
