import os
from typing import Optional, Callable

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import is_image_file


def load_images(root_path: str):
    """return paths for all images found in folder and subfolders"""
    images = []
    assert os.path.isdir(root_path), '%s is not a valid directory' % root_path

    for root, _, fnames in sorted(os.walk(root_path)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class StereoVisionDataset(VisionDataset):
    """
    A dataset class for paired image datasets.
    It assumes that the directory '/path/to/data/train' contains images that store a
    left and right frame for stereo vision.
    """

    def __init__(self, root: str, transforms: Optional[Callable] = None):
        """
        Initialize this dataset class.
        """
        super(StereoVisionDataset, self).__init__(root, transforms=transforms)
        self.img_paths = sorted(load_images(self.root))  # get image paths

    def __getitem__(self, index: int):
        """
        Return a single data point.
        :param index: an integer in [0, len(dataset)[ for data indexing
        """
        # read an image given a random integer index
        img_path = self.img_paths[index]
        frame = Image.open(img_path).convert('RGB')
        # split AB image into A and B
        w, h = frame.size
        w2 = int(w / 2)
        left = frame.crop((0, 0, w2, h))
        right = frame.crop((w2, 0, w, h))

        if self.transforms:
            left, right = self.transforms(left), self.transforms(right)

        return left, right

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)


if __name__ == '__main__':
    def _main():
        dataset = StereoVisionDataset('./data')
        print(len(dataset))
        x, y = dataset[0].values()
        x.show()
        y.show()
    _main()
