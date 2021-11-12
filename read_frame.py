import os
from random import randint
import argparse

import cv2


def extract_frames(
        vid_path: str,
        output_path: str,
        output_name: str,
        start_idx: int = 5000,
        min_offs: int = 2,
        max_offs: int = 25
):
    """
    Extract frames from video file. Randomizes offset between subsequent extracted frames

    :param vid_path: path to video file
    :param output_path: directory to store results in
    :param output_name: base file name for outputs
    :param start_idx: start frame for extraction
    :param min_offs: min offset (#frames) between subsequent saved frames
    :param max_offs: max offset between subsequent frames
    :return: None
    """
    vidcap = cv2.VideoCapture(vid_path)

    frame_idx = start_idx
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = vidcap.read()
    base_name = os.path.join(output_path, output_name)

    while success:
        cv2.imwrite(f"{base_name}_frame_{frame_idx}.jpg", frame)
        frame_idx += randint(min_offs, max_offs)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = vidcap.read()

    vidcap.release()


if __name__ == '__main__':
    def _main():
        parser = argparse.ArgumentParser(description="Extract movie frames")

        parser.add_argument('input', type=str, help='Input movie file')
        parser.add_argument('--output_path', type=str, default='.', help='output directory')
        parser.add_argument('--base_name', tupe=str, default='movie', help='base output file name')
        parser. add_argument('--min_offset', type=int, default=2, help='min offset between frames')
        parser.add_argument('--max_offset', type=int, default=25, help='max offset between frames')
        parser.add_argument('--start_frame', type=int, default=5000, help='first frame to extract')

        args = parser.parse_args()
        assert args.min_offset <= args.max_offset, 'min offset must be <= max offset'

        extract_frames(
            vid_path=args.input,
            output_path=args.output_path,
            output_name=args.base_name,
            start_idx=args.start_frame,
            min_offs=args.min_offset,
            max_offs=args.max_offset
        )
