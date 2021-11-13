import multiprocessing
import os
from functools import partial
from random import randint
import argparse

import cv2


def extract_frames(
        vid_path: str,
        output_path: str,
        output_name: str,
        start_idx: int = 50,
        end_idx: int = -1,
        min_offs: int = 2,
        max_offs: int = 25
):
    """
    Extract frames from video file. Randomizes offset between subsequent extracted frames.
    Extracted frames will be stored as 'output_path/output_name_frame_IDX.jpg'

    :param vid_path: path to video file
    :param output_path: directory to store results in
    :param output_name: base file name for outputs
    :param start_idx: start frame for extraction
    :param end_idx: final frame for extraction
    :param min_offs: min offset (#frames) between subsequent saved frames
    :param max_offs: max offset between subsequent frames
    :return: None
    """
    vidcap = cv2.VideoCapture(vid_path)

    frame_idx = start_idx
    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert end_idx < n_frames, 'illegal end frame'

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = vidcap.read()
    base_name = os.path.join(output_path, output_name)

    while success:
        cv2.imwrite(f"{base_name}_frame_{frame_idx}.jpg", frame)
        frame_idx += randint(min_offs, max_offs)
        if frame_idx > end_idx:
            break
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = vidcap.read()

    vidcap.release()
    return 1


def multi_extract(
        vid_path: str,
        output_path: str,
        output_name: str,
        start_idx: int = 50,
        end_idx: int = -1,
        min_offs: int = 2,
        max_offs: int = 25
):
    f = partial(
        extract_frames,
        vid_path=vid_path,
        output_path=output_path,
        output_name=output_name,
        min_offs=min_offs,
        max_offs=max_offs)

    n_procs = multiprocessing.cpu_count()
    vidcap = cv2.VideoCapture(vid_path)
    max_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vidcap.release()

    if end_idx == -1:
        end_idx = max_frames - 1
    n_frames = end_idx - start_idx

    frames_per_proc = n_frames // n_procs
    args = []
    idx = start_idx

    for i in range(n_procs):
        next_idx = end_idx if i == (n_procs - 1) else idx + frames_per_proc
        args.append((idx, next_idx))
        idx = next_idx
    print(args)

    with multiprocessing.Pool(processes=n_procs) as pool:
        results = [pool.apply_async(f, kwds={'start_idx':args[i][0], 'end_idx':args[i][1]}) for i in range(n_procs)]
        [result.wait() for result in results]
        print(results[-1].get())


if __name__ == '__main__':
    def _main():
        parser = argparse.ArgumentParser(description="Extract movie frames")

        parser.add_argument('input', type=str, help='Input movie file')
        parser.add_argument('--output_path', type=str, default='./data', help='output directory')
        parser.add_argument('--base_name', type=str, default='movie', help='base output file name')
        parser. add_argument('--min_offset', type=int, default=2, help='min offset between frames')
        parser.add_argument('--max_offset', type=int, default=25, help='max offset between frames')
        parser.add_argument('--start_frame', type=int, default=5000, help='first frame to extract')
        parser.add_argument('--end_frame', type=int, default=-1, help='last frame to extract')

        args = parser.parse_args()
        assert args.min_offset <= args.max_offset, 'min offset must be <= max offset'

        multi_extract(
            vid_path=args.input,
            output_path=args.output_path,
            output_name=args.base_name,
            start_idx=args.start_frame,
            end_idx=args.end_frame,
            min_offs=args.min_offset,
            max_offs=args.max_offset
        )

    _main()

