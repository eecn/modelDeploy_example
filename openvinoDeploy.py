import cv2
import numpy as np
from openvino.runtime import Core

ie = Core()
devices = ie.available_devices

for device in devices:
    device_name = ie.get_property(device_name=device, name="FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")

onnx_model_path = "srcnn3.onnx"
model_onnx = ie.read_model(model=onnx_model_path)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")

from openvino.offline_transformations import serialize

serialize(model=model_onnx, model_path="exported_onnx_model.xml",
          weights_path="exported_onnx_model.bin")

model_xml = "exported_onnx_model.xml"
model = ie.read_model(model=model_xml)
model.input(0).any_name
model.output(0).any_name
input_layer = model.input(0)
output_layer = model.output(0)
input_layer2 = model.input(1)


print(f"input precision: {input_layer.element_type}")
print(f"input shape: {input_layer.shape}")
print(f"output precision: {output_layer.element_type}")
# print(f"output shape: {output_layer.shape}")

compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

input_img = cv2.imread('face.png').astype(np.float32)
# print(input_img.ndim)
# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

input_factor = np.array([1, 1, 4, 4], dtype=np.float32)

result = compiled_model([input_img,input_factor])[output_layer]
print(result.shape)
result = np.squeeze(result, 0)
result = np.clip(result, 0, 255)
result = np.transpose(result, [1, 2, 0]).astype(np.uint8)
cv2.imwrite("face_openvino.png", result)