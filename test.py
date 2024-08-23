import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from flask import Flask

app = Flask(__name__)



def what():
# select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device;

@app.route('/')
def run_pytorch_function():
    res = what()
    return f"using: {res}";

if __name__ == '__main__':
    app.run(host='0.0.0.0')
