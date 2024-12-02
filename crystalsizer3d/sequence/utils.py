import glob
from argparse import Namespace
from pathlib import Path
from typing import List, Tuple

from crystalsizer3d.args.sequence_fitter_args import SequenceFitterArgs


def get_image_paths(args: Namespace | SequenceFitterArgs, load_all: bool = False) -> List[Tuple[int, Path]]:
    """
    Load the images defined in the args.
    """
    pathspec = str(args.images_dir.absolute()) + '/*.' + args.image_ext
    all_image_paths = sorted(glob.glob(pathspec))
    image_paths = [(idx, Path(all_image_paths[idx])) for idx in range(
        args.start_image,
        len(all_image_paths) if args.end_image == -1 else args.end_image,
        args.every_n_images if hasattr(args, 'every_n_images') and args.every_n_images > 1 and not load_all else 1
    )]
    return image_paths
