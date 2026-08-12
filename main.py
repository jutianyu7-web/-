import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
LEARNING_RATE = 0.04
LEARNING_ROUNDS=15
def traindata_read():
    labels_file = "artifacts/labels.npy"
    images_file = "artifacts/images.npy"


    if os.path.exists(labels_file) and os.path.exists(images_file):
        # 缓存文件存在：直接读取
        print("正在读取已经处理好的数据……")
    else:
        # 缓存文件不存在：第一次读取并处理 CSV
        print("第一次运行，正在处理 CSV……")
        data = pd.read_csv("data/train.csv.zip")
        # 第一列转换为整数标签
        labels = data.iloc[:, 0].to_numpy(dtype=np.int64)
        # 其余列转换为像素，并变成 28×28 图片
        images = data.iloc[:, 1:].to_numpy(dtype=np.uint8)
        images = images.reshape(-1, 28, 28)
        # 确保保存文件夹存在
        os.makedirs("artifacts", exist_ok=True)
        # 保存处理结果
        np.save(labels_file, labels)
        np.save(images_file, images)
        print("处理完成，数据已经保存。")

def change_1(image):
    # 先转换成浮点数，再除以 255
    images_normalized = image.astype(np.float32) / 255.0
    return images_normalized

def make_batch(X,y,bench_size,shuffle=True, rng=None):
    assert len(X) == len(y)
    indices = np.arange(len(X))
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)
    for start in range(0,len(X),bench_size):
        end=start+bench_size
        bench_result=indices[start:end]
        X_train = X[bench_result]
        y_train = y[bench_result]
        yield X_train,y_train

def linear_forward(trainx,W1,b1):
    # assert trainx.ndim == 2
    # assert trainx.shape[1] == 784
    # assert W1.shape == (784, 128)
    # assert b1.shape == (128,)
    Z1 = trainx @ W1 + b1


    return Z1



def softmax(image_data):
    # 每张图片分别找最大分数，结果形状为 (B, 1)
    MAX_data = np.max(image_data, axis=1, keepdims=True)

    # 每一行分别减去自己的最大值
    shifted_data = image_data - MAX_data
    # 计算指数
    exp_data = np.exp(shifted_data)
    # 每一行的10个指数分别求和
    sum_exp = np.sum(exp_data, axis=1, keepdims=True)
    # 得到概率
    probability = exp_data / sum_exp

    return probability

def cross_entropy(probability, trainy):
    batch_size = len(trainy)

    correct_probability = probability[
        np.arange(batch_size),
        trainy
    ]

    loss = -np.mean(
        np.log(correct_probability + 1e-12)
    )

    return loss

def linear_backward(trainx,A1,probability,trainy,W1,b1,W2,b2,Z1):
    B = trainx.shape[0]
    dZ2 = probability.copy()
    dZ2[np.arange(B), trainy] -= 1.0
    dZ2 /= B

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)
    dA1 = dZ2 @ W2.T
    dZ1=ReLU_backward(dA1,Z1)
    dW1 = trainx.T @ dZ1
    db1 = np.sum(dZ1, axis=0)



    # assert dZ.shape == probability.shape
    # assert dW.shape == W.shape
    # assert db.shape == b.shape
    # assert np.all(np.isfinite(dW))
    # assert np.all(np.isfinite(db))

    return dW1,db1,dW2,db2

def ReLU(x):
    return np.maximum(0, x)#用于逐元素比较

def ReLU_backward(dA1, Z1):
    dZ1 = dA1.copy()
    dZ1[Z1 <= 0] = 0
    return dZ1

def photo_read(photo_path):
    # 手机照片通常是白底黑字，需要转换成训练数据的黑底白字
    with Image.open(photo_path) as original_image:
        image = ImageOps.exif_transpose(original_image).convert("L")
    image = ImageOps.autocontrast(image)
    image = ImageOps.invert(image)

    # 找出数字区域并去掉照片周围的空白
    image_array = np.array(image)
    position = np.argwhere(image_array > 30)
    if len(position) == 0:
        raise ValueError("照片中没有检测到数字")

    top, left = position.min(axis=0)
    bottom, right = position.max(axis=0)
    image = image.crop((left, top, right + 1, bottom + 1))

    # 缩放后放到 28×28 黑色画布中央
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    processed_image = Image.new("L", (28, 28), 0)
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2
    processed_image.paste(image, (x, y))

    photo_x = np.array(processed_image, dtype=np.float32)
    photo_x = photo_x.reshape(1, 784) / 255.0
    return photo_x, processed_image

def predict_photo(photo_x, W1, b1, W2, b2):
    Z1 = linear_forward(photo_x, W1, b1)
    A1 = ReLU(Z1)
    Z2 = linear_forward(A1, W2, b2)
    probability = softmax(Z2)
    result = np.argmax(probability, axis=1)[0]
    confidence = probability[0, result]
    return result, confidence


def main():
    """_______________准备数据_______________"""
    #X是图像，y是答案
    traindata_read()
    images = np.load("artifacts/images.npy")
    labels = np.load("artifacts/labels.npy")
    X = images.reshape(42000, 784)
    X = change_1(X)
    y = labels.astype(np.int64)
    indices = np.arange(42000)
    # 使用固定的随机种子打乱索引
    rng = np.random.default_rng(seed=42)
    rng.shuffle(indices)
    # 4. 按照 80% / 20% 切分索引
    split_position = int(42000 * 0.8)
    train_indices = indices[:split_position]
    val_indices = indices[split_position:]
    # 5. 使用同一组索引切分 X 和 y
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    #随机选取64个元素准备训练
    #参数W(784,10)，偏执b(10,)，得到结果Z(64,10)
    # 单独创建模型参数的随机数生成器
    model_rng = np.random.default_rng(seed=42)
    # W 使用较小的随机数初始化
    W1= model_rng.normal(
        loc=0.0,
        scale=0.01,
        size=(784, 128)
    ).astype(np.float32)
    W2 = model_rng.normal(
        loc=0.0,
        scale=0.01,
        size=(128, 10)
    ).astype(np.float32)

    # b 从 0 开始
    b1= np.zeros(128, dtype=np.float32)
    b2= np.zeros(10, dtype=np.float32)
    # 固定训练时的随机数生成器：每轮顺序不同，但每次运行结果一致
    train_rng = np.random.default_rng(seed=42)
    val_loss_history = []
    val_accuracy_history = []




    for i in range(LEARNING_ROUNDS):
        for trainx,trainy in make_batch(X_train,y_train,64,rng=train_rng):
            Z1=linear_forward(trainx,W1,b1)
            A1=ReLU(Z1)
            Z2=linear_forward(A1,W2,b2)
            probability = softmax(Z2)
            dw1,db1,dw2,db2=linear_backward(trainx,A1,probability,trainy,W1,b1,W2,b2,Z1)
            W1-=LEARNING_RATE*dw1
            b1-=LEARNING_RATE*db1
            W2-=LEARNING_RATE*dw2
            b2-=LEARNING_RATE*db2

        # 一整轮训练结束后，计算整个训练集的结果
        # train_Z = linear_forward(X_train,W,b)
        # train_probability = softmax(train_Z)
        # train_loss = cross_entropy(train_probability,y_train)
        # train_result = np.argmax(train_probability,axis=1)
        # train_accuracy = np.mean(train_result == y_train)

        # 使用没有参加训练的验证集检查模型效果
        val_Z1 = linear_forward(X_val, W1, b1)
        val_A1 = ReLU(val_Z1)
        val_Z2 = linear_forward(val_A1, W2, b2)
        val_probability = softmax(val_Z2)
        val_loss = cross_entropy(val_probability,y_val)
        val_result = np.argmax(val_probability,axis=1)
        val_accuracy = np.mean(val_result == y_val)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        print(f"第{i+1}轮")
        # print("训练集 loss：", train_loss)
        # print("训练集 accuracy：", train_accuracy)
        print("验证集 loss：", val_loss)
        print("验证集 accuracy：", val_accuracy)

    photo_path = input("请输入手写数字照片路径（直接回车跳过）：").strip()
    photo_path = photo_path.strip('"').strip("'").replace("\\ ", " ")
    if photo_path:
        try:
            photo_x, processed_image = photo_read(photo_path)
            result, confidence = predict_photo(photo_x, W1, b1, W2, b2)
            print("照片识别结果：", result)
            print("模型置信度：", confidence)

            plt.figure("Photo prediction")
            plt.imshow(processed_image, cmap="gray")
            plt.title(f"Prediction: {result}")
            plt.axis("off")
        except (FileNotFoundError, OSError, ValueError) as error:
            print("照片读取失败：", error)

    rounds = range(1, LEARNING_ROUNDS + 1)
    plt.figure("Training curves", figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rounds, val_loss_history)
    plt.xlabel("Round")
    plt.ylabel("Validation loss")
    plt.title("Loss curve")

    plt.subplot(1, 2, 2)
    plt.plot(rounds, val_accuracy_history)
    plt.xlabel("Round")
    plt.ylabel("Validation accuracy")
    plt.title("Accuracy curve")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
