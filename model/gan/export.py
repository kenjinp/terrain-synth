import torch
import datetime


def export(model, input, filename=f"{datetime.datetime.now().strftime('%m-%d-%y_%H:%M:%S')}.onnx"):
    with torch.inference_mode(), torch.cuda.amp.autocast():
        # torch.onnx.dynamo_export(model, input).save(filename)
        torch.onnx.export(model,
                          (input),
                          filename,
                          verbose=False
                          )
