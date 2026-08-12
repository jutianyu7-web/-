import numpy as np
label=np.load("artifacts/labels.npy")
image=np.load("artifacts/images.npy")
np.set_printoptions(linewidth=200)
print(label[1])
print(image[1])
"""import numpy as np


def make_batches(X, y, batch_size, shuffle=True, rng=None):
    # 确保图片数量和标签数量一致
    assert len(X) == len(y)

    # 生成样本索引：0, 1, 2, ..., N-1
    indices = np.arange(len(X))

    # 根据索引进行打乱
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)

    # 每次取出 batch_size 个索引
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]

        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        yield X_batch, y_batch"""