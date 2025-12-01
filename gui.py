import argparse
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageDraw, ImageTk

from core.layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense
from core.losses import SoftmaxCrossEntropy
from core.model import Sequential


# ============================
# MODELS
# ============================

def build_mnist():
    layers=[
        Conv2D(1,8,3,padding=1),
        ReLU(),
        MaxPool2D(2,2),

        Conv2D(8,16,3,padding=1),
        ReLU(),
        MaxPool2D(2,2),

        Flatten(),
        Dense(16*7*7,64),
        ReLU(),
        Dense(64,10)
    ]
    return Sequential(layers, SoftmaxCrossEntropy())

def build_shapes():
    layers=[
        Conv2D(1,8,3,padding=1),
        ReLU(),
        MaxPool2D(2,2),

        Conv2D(8,16,3,padding=1),
        ReLU(),
        MaxPool2D(2,2),

        Flatten(),
        Dense(16*8*8,32),
        ReLU(),
        Dense(32,2)
    ]
    return Sequential(layers, SoftmaxCrossEntropy())


# ============================
# PREPROCESS
# ============================

def preprocess_mnist(arr):
    thresh = arr > 0.1
    if np.sum(thresh)==0:
        return np.zeros((28,28), np.float32)

    ys, xs = np.where(thresh)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    digit = arr[ymin:ymax+1, xmin:xmax+1]

    h,w = digit.shape
    scale = 20.0 / max(h,w)
    nh, nw = max(1,int(h*scale)), max(1,int(w*scale))

    img = Image.fromarray((digit*255).astype(np.uint8)).resize((nw,nh), Image.NEAREST)
    canvas = Image.new("L",(28,28),0)
    canvas.paste(img,((28-nw)//2,(28-nh)//2))

    return np.array(canvas, np.float32)/255.0

def preprocess_shape(path):
    img = Image.open(path).convert("L").resize((32,32), Image.NEAREST)
    return np.array(img, np.float32)/255.0



# ============================
# MODERN GUI APP
# ============================

class App:
    def __init__(self, ckpt_digit, ckpt_shape):
        self.digit_model = build_mnist()
        self.shape_model = build_shapes()
        self.digit_model.load(ckpt_digit)
        self.shape_model.load(ckpt_shape)
        self.shape_classes = ["rectangle", "circle"]

        # ROOT UI
        self.root = tk.Tk()
        self.root.title("Deep Vision — Digit & Shape Recognition")
        self.root.configure(bg="#ECECEC")

        # UI STYLE
        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 11), padding=6)
        style.configure("TLabel", background="#ECECEC", font=("Segoe UI", 12))

        # Mode selection
        self.mode = tk.StringVar(value="digit")
        mode_frame = ttk.Frame(self.root)
        mode_frame.pack(pady=10)

        ttk.Radiobutton(mode_frame, text="Digit Draw", variable=self.mode, value="digit", command=self.update_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Shape Upload", variable=self.mode, value="shape", command=self.update_mode).pack(side=tk.LEFT, padx=5)

        # Canvas / Preview Frame
        self.display_frame = tk.Frame(self.root, bg="#D8D8D8", width=300, height=300)
        self.display_frame.pack(pady=10)

        # Canvas (digit mode)
        self.canvas_size = 280
        self.canvas = tk.Canvas(self.display_frame, width=self.canvas_size, height=self.canvas_size, bg="black", highlightthickness=2, highlightbackground="#444")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.img = Image.new("L",(self.canvas_size,self.canvas_size),0)
        self.draw = ImageDraw.Draw(self.img)

        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=8)

        ttk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        self.upload_btn = ttk.Button(btn_frame, text="Upload Image", command=self.load_image)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        self.upload_btn["state"] = "disabled"

        ttk.Button(btn_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)

        # Output Label
        self.result_label = ttk.Label(self.root, text="Prediction: ?", font=("Segoe UI", 16))
        self.result_label.pack(pady=10)

        self.shape_arr = None

    def update_mode(self):
        if self.mode.get()=="digit":
            self.upload_btn["state"] = "disabled"
            self.canvas["state"] = "normal"
            self.clear()
        else:
            self.upload_btn["state"] = "normal"
            self.canvas["state"] = "disabled"
            self.result_label.config(text="Upload an image to start.")

    def paint(self, e):
        if self.mode.get()!="digit":
            return
        r = 8
        self.canvas.create_oval(e.x-r,e.y-r,e.x+r,e.y+r, fill="white", outline="white")
        self.draw.ellipse([e.x-r,e.y-r,e.x+r,e.y+r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.img = Image.new("L",(self.canvas_size,self.canvas_size),0)
        self.draw = ImageDraw.Draw(self.img)
        self.shape_arr = None
        self.result_label.config(text="Prediction: ?")

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        if not path:
            return

        self.shape_arr = preprocess_shape(path)

        # Preview 150x150
        preview = Image.open(path).convert("L").resize((200,200))
        preview_tk = ImageTk.PhotoImage(preview)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=preview_tk)
        self.canvas.image = preview_tk

        self.result_label.config(text="Image loaded. Click Predict.")

    def predict(self):
        if self.mode.get()=="digit":
            arr = np.array(self.img, np.float32)/255.0
            arr = preprocess_mnist(arr)
            x = arr[np.newaxis,np.newaxis,:,:]

            logits = self.digit_model.forward(x,train=False)[0]
            ex = np.exp(logits-logits.max())
            pred = int(np.argmax(logits))
            conf = ex.max()/ex.sum()

            self.result_label.config(text=f"Prediction: {pred}   ({conf:.3f})")

        else:
            if self.shape_arr is None:
                self.result_label.config(text="Please upload an image first!")
                return

            x = self.shape_arr[np.newaxis,np.newaxis,:,:]
            logits = self.shape_model.forward(x,train=False)[0]
            ex = np.exp(logits-logits.max())
            pred = int(np.argmax(logits))
            conf = ex.max()/ex.sum()

            cls = self.shape_classes[pred]
            self.result_label.config(text=f"Prediction: {cls}   ({conf:.3f})")

    def run(self):
        self.root.mainloop()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_digit", required=True)
    ap.add_argument("--ckpt_shape", required=True)
    args = ap.parse_args()

    App(args.ckpt_digit, args.ckpt_shape).run()


if __name__ == "__main__":
    main()

#python gui.py --ckpt_digit checkpoints/mnist_best.npz --ckpt_shape checkpoints/shapes_best.npz