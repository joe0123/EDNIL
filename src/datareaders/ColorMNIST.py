import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets


def ColorMNIST(data_root, label_noise=0.2, val_ratio=0,
            train_color_noise=0.15, test_color_noises=np.arange(0.1, 1, 0.2)):
    def _make_environment(images, shapes, e):
        # NOTE: low e indicates a spurious correlation from color to (noisy) label
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()

        def torch_xor(a, b):
            assert a.shape == b.shape
            return (a - b).abs() # Assumes both inputs are either 0 or 1

        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability label_noise
        shapes = (shapes < 5).float()
        labels = torch_xor(shapes, torch_bernoulli(p=label_noise, size=len(shapes)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(p=e, size=len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
        images = (images.float() / 255.).reshape(images.shape[0], -1)
        labels = labels.reshape(-1)
        assert images.shape[0] == labels.shape[0]
        return images, labels
    
    mnist = datasets.MNIST(data_root, train=True, download=True)
    train_inds = np.random.permutation(50000)
    mnist_train = (mnist.data[:50000][train_inds], mnist.targets[:50000][train_inds])
    mnist_test = (mnist.data[50000:], mnist.targets[50000:])
    
    train_valid_data = _make_environment(images=mnist_train[0], 
                                         shapes=mnist_train[1], 
                                         e=train_color_noise)
    train_mask = np.random.choice([False, True],
                                  size=train_valid_data[0].shape[0], 
                                  p=[val_ratio, 1 - val_ratio])
    train_features = train_valid_data[0][train_mask]
    train_labels = train_valid_data[1][train_mask]
    train_envs = [{'X': train_features, 
                   'y': train_labels.reshape(-1, 1).float()}]
    
    valid_mask = ~train_mask
    if valid_mask.sum() > 0:
        valid_features = train_valid_data[0][valid_mask]
        valid_labels = train_valid_data[1][valid_mask]
        valid_envs = [{'X': valid_features, 
                       'y': valid_labels.reshape(-1, 1).float()}]
    else:
        valid_envs = []

    test_envs = []
    for test_color_noise in test_color_noises:
        test_data = _make_environment(images=mnist_test[0], 
                                      shapes=mnist_test[1], 
                                      e=test_color_noise)
        test_features, test_labels = test_data
        test_envs.append({'X': test_features, 
                          'y': test_labels.reshape(-1, 1).float()})
    
    return train_envs, valid_envs, test_envs
