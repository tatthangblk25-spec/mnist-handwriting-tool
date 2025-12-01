import os, argparse, json, numpy as np
from core.layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense
from core.losses import SoftmaxCrossEntropy
from core.model import Sequential
from core.optim import SGD
from core.utils import accuracy, batch_iter
from datasets.mnist import load_mnist
from datasets.shapes import gen_shapes_dataset


def build_model(task):
    # Xây kiến trúc CNN tùy theo task
    if task == "mnist":
        layers = [
            Conv2D(1, 8, 3, padding=1),
            ReLU(),
            MaxPool2D(2, 2),

            Conv2D(8, 16, 3, padding=1),
            ReLU(),
            MaxPool2D(2, 2),

            Flatten(),  # Chuyển sang vector để đưa vào Dense
            Dense(16 * 7 * 7, 64),
            ReLU(),
            Dense(64, 10),  # Output layer cho 10 lớp MNIST
        ]
        loss = SoftmaxCrossEntropy()
        return Sequential(layers, loss)

    elif task == "shapes":
        layers = [
            Conv2D(1, 8, 3, padding=1), 
            ReLU(),
            MaxPool2D(2, 2),

            Conv2D(8, 16, 3, padding=1),  
            ReLU(),
            MaxPool2D(2, 2),

            Flatten(),
            Dense(16 * 8 * 8, 32),  
            ReLU(),
            Dense(32, 2),  # 2 lớp: circle hoặc rectangle
        ]
        loss = SoftmaxCrossEntropy()
        return Sequential(layers, loss)

    else:
        raise ValueError("Unknown task")


def main():
    # Parser tham số dòng lệnh
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mnist", "shapes"], required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=6000, help="for shapes")
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = build_model(args.task)  # Tạo model

    # Load dataset tùy task
    if args.task == "mnist":
        (X_tr, y_tr), (X_te, y_te) = load_mnist()
        val_n = int(len(X_te) * 0.5)
        X_val, y_val = X_te[:val_n], y_te[:val_n]  # Lấy nửa test làm validation
        X_te2, y_te2 = X_te[val_n:], y_te[val_n:]  # Nửa còn lại để test cuối
    else:
        X, y, _ = gen_shapes_dataset(n_samples=args.samples, seed=args.seed)
        n = int(len(X) * (1 - args.val_split))
        X_tr, y_tr = X[:n], y[:n]
        X_val, y_val = X[n:], y[n:]
        X_te2, y_te2 = X[n:], y[n:]

    # Optimizer
    optim = SGD(model.params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    os.makedirs("checkpoints", exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        losses, accs = [], []

        # Train loop
        for xb, yb in batch_iter(X_tr, y_tr, args.batch, shuffle=True, rng=rng):
            logits = model.forward(xb, train=True)  # Forward
            loss, grad = model.compute_loss_and_grad(logits, yb)  # Loss + grad
            model.backward(grad)  # Backprop
            optim.step()  # Update weights

            losses.append(loss)
            accs.append(accuracy(logits, yb))

        # Validation
        logits_val = model.forward(X_val, train=False)
        val_acc = accuracy(logits_val, y_val)

        print(f"Epoch {epoch}: loss={np.mean(losses):.4f} | train_acc={np.mean(accs):.4f} | val_acc={val_acc:.4f}")

        # Lưu checkpoint tốt nhất
        if val_acc > best_acc:
            best_acc = val_acc
            path = f"checkpoints/{args.task}_best.npz"
            model.save(path)
            print(f"Saved best to {path}")

    # Test cuối
    logits_test = model.forward(X_te2, train=False)
    te_acc = accuracy(logits_test, y_te2)
    print(f"Test accuracy: {te_acc:.4f}")


if __name__ == "__main__":
    main()
