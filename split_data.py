import glob
import os
import shutil
from random import randint


def split_data(source_path: str, dest_path: str, n_samples: int = 2000, block_size: int = 100):
    assert block_size < n_samples, 'invalid block size'
    source_files = list(sorted(glob.glob(os.path.join(source_path, '*.jpg'))))
    result = []
    while len(result) < n_samples:
        n_files = min(block_size, n_samples - len(result))
        idx = randint(0, len(source_files) - n_files)
        # sample blocks of subsequent frames to avoid v similar frames in train and test
        result += source_files[idx:idx + n_files]
        del source_files[idx:idx + n_files]

    for file in result:
        print(f"Moving {file} to {dest_path}")
        shutil.move(file, dest_path)
    print(f"Moved {len(result)} files")


if __name__ == '__main__':
    split_data('./data/train', './data/test')

