# Converts to ONNX/TorchScript + quantizes

class Export:
    def __init__(self, model):
        self.model = model

    # Export with torch.quantization or ONNX
    def export(self):
        pass

