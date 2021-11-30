from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class DisentangledDataLoader(ABC):
    @abstractmethod
    def get_train_dataset(self):
        pass

    @abstractmethod
    def get_validation_dataset(self):
        pass

    @abstractmethod
    def get_test_dataset(self):
        pass

    @abstractmethod
    def plot_image(self, image, labels, filename=None):
        pass

    def get_batch_size(self):
        return self.batch_size

    def get_data_loader(self, dataset_split):
        if dataset_split == 'train':
            dataset = self.get_train_dataset()
        elif dataset_split == 'val':
            dataset = self.get_validation_dataset()
        elif dataset_split == 'test':
            dataset = self.get_test_dataset()
        else:
            raise ValueError('Dataset split must be train, val, or test!')
        return DataLoader(dataset, shuffle=False, num_workers=0,  # for determinism
                          batch_size=self.get_batch_size())

    def demo_data(self, split, num_examples=3):
        dl = self.get_data_loader(split)
        batches = 0
        while num_examples > batches * self.get_batch_size():
            example = iter(dl).next()
            imgs = example[0]
            labels = example[1:]
            for i in range(num_examples):
                self.plot_image(imgs[i], [l[i] for l in labels])
            batches += 1
