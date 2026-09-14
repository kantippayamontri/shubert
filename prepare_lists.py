import argparse
import glob
import gzip
import os
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="dataset root containing <split>/raw_videos/")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="directory for .list files")
    parser.add_argument("--splits", type=str, required=True,
                        help="comma-separated splits, e.g. val,test")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for split in args.splits.split(","):
        videos = sorted(glob.glob(os.path.join(args.root, split, "raw_videos", "*.mp4")))
        out_file = os.path.join(args.out_dir, f"how2sign_{split}.list")
        with gzip.GzipFile(out_file, "wb") as f:
            f.write(pickle.dumps(videos, protocol=0))
        print(f"{split}: {len(videos)} videos -> {out_file}")


if __name__ == "__main__":
    main()