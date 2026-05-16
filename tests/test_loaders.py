from data.loaders import get_loaders

def test_cifar10_shapes():
    train, val, test = get_loaders(dataset="cifar10", batch_size=64)
    images, labels = next(iter(train))
    assert images.shape == (64, 3, 32, 32)
    assert labels.shape == (64,)
    print('CIFAR-10 passed:', images.shape)

if __name__ == '__main__':
    test_cifar10_shapes()