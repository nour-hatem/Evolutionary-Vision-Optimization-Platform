import torchvision.transforms as T
from data.stats import get_stats


class CIFAR10Transforms:
    def __init__(self, mean, std, augment=True):
        self.mean = mean
        self.std = std
        self.augment = augment

    def _normalize(self):
        return T.Normalize(self.mean, self.std)

    def train(self):
        if self.augment:
            return T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                self._normalize()
            ])
        return T.Compose([
            T.ToTensor(),
            self._normalize()
        ])

    def test(self):
        return T.Compose([
            T.ToTensor(),
            self._normalize()
        ])


def get_transforms(augment=True):
    stats = get_stats()
    mean = stats['mean']
    std = stats['std']

    obj = CIFAR10Transforms(mean, std, augment=augment)
    return obj.train(), obj.test()