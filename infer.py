import os, argparse, numpy as np
from datasets.utils import load_image_gray
from core.layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense
from core.losses import SoftmaxCrossEntropy
from core.model import Sequential

def build(task):
    if task=="mnist":
        layers=[Conv2D(1,8,3,padding=1),
                ReLU(),
                MaxPool2D(2,2),
                
                Conv2D(8,16,3,padding=1),
                ReLU(),
                MaxPool2D(2,2),
                
                Flatten(),
                Dense(16*7*7,64),
                ReLU(),
                Dense(64,10)]
        classes=[str(i) for i in range(10)]
        size=(28,28)
    else:
        layers=[Conv2D(1,8,3,padding=1),
                ReLU(),
                MaxPool2D(2,2),

                Conv2D(8,16,3,padding=1),
                ReLU(),
                MaxPool2D(2,2),
                Flatten(),
                
                Dense(16*8*8,32),
                ReLU(),
                Dense(32,2)]
        classes=["rectangle","circle"]
        size=(32,32)
    return Sequential(layers, SoftmaxCrossEntropy()), classes, size

def predict_file(model, size, path):
    x = load_image_gray(path, size=size)
    x = x[np.newaxis, ...]
    logits = model.forward(x, train=False)[0]
    return logits

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mnist","shapes"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="image file or folder")
    args=ap.parse_args()

    model, classes, size = build(args.task)
    model.load(args.ckpt)

    paths=[]
    if os.path.isdir(args.input):
        for fn in os.listdir(args.input):
            if fn.lower().endswith((".png",".jpg",".jpeg",".bmp")):
                paths.append(os.path.join(args.input, fn))
    else:
        paths=[args.input]

    import numpy as np
    for p in paths:
        logits = predict_file(model, size, p)
        pred = int(np.argmax(logits))
        conf = float(np.max(np.exp(logits-logits.max()))/np.sum(np.exp(logits-logits.max())))
        print(f"{p}: {classes[pred]} (conf={conf:.3f})")

if __name__=="__main__":
    main()