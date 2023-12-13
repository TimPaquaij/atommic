# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse
import json
from pathlib import Path
import numpy as np
import random

def generate_fold(filenames):
    """Generate a train, val and test set from a list of filenames"""
    data_parent_dir = Path(filenames[0]).parent

    # Path to str
    filenames = [str(filename) for filename in filenames]

    # keep only the filename, so drop the "-t1c.nii.gz", "-t1n.nii.gz", "-t2f.nii.gz", or "-t2w.nii.gz"
    filenames = [filename.split("/")[-1] for filename in filenames]
    # keep only the unique filenames
    filenames = np.unique(filenames)

    # shuffle the filenames
    random.shuffle(filenames)

    # split the filenames into train and val with 80% and 20% respectively
    train_fnames = np.array(filenames[: int(len(filenames) * 0.8)]).tolist()
    # remove train filenames from all filenames
    filenames = np.setdiff1d(filenames, train_fnames)
    # since we have already removed the train filenames, we can use the remaining filenames as val
    val_fnames = filenames.tolist()

    # set full path
    train_fnames = [str(data_parent_dir / filename) for filename in train_fnames]
    val_fnames = [str(data_parent_dir / filename) for filename in val_fnames]

    return train_fnames, val_fnames

def main(args):
    if args.data_type == "raw":
        data_type = "files_recon_calib-24"
    else:
        data_type = "image_files"

    # remove "annotations/v1.0.0/" from args.annotations_path and add "files_recon_calib-24" to get the raw_data_path
    raw_data_path = Path(args.annotations_path).parent.parent / data_type

    # get train.json, val.json and test.json filenames from args.annotations_path
    annotations_sets = list(Path(args.annotations_path).iterdir())

    if args.nfold is not None:
        for annotation_set in annotations_sets:
            set_name = Path(annotation_set).name
            print(set_name)
            if set_name =='train.json':
                print('go')
                with open(annotation_set, "r", encoding="utf-8") as f:
                    annotation_set = json.load(f)
                filenames = [f'{raw_data_path}/{image["file_name"]}' for image in annotation_set["images"]]
                folds = [generate_fold(filenames) for _ in range(args.nfold)]

                # create a directory to store the folds
                output_path = Path(args.output_path) / "folds"
                output_path.mkdir(parents=True, exist_ok=True)

                # write each fold to a json file
                for i, fold in enumerate(folds):
                    train_set, val_set = fold

                    # write the train, val and test filenames to a json file
                    with open(output_path / f"{data_type}_fold_{i}_train.json", "w", encoding="utf-8") as f:
                        json.dump(train_set, f)
                    with open(output_path / f"{data_type}_fold_{i}_val.json", "w", encoding="utf-8") as f:
                        json.dump(val_set, f)
            if set_name =='test.json':
                with open(annotation_set, "r", encoding="utf-8") as f:
                    annotation_set = json.load(f)

                # read the "images" key and for every instance get the "file_name" key
                filenames = [f'{raw_data_path}/{image["file_name"]}' for image in annotation_set["images"]]

                # create a directory to store the folds
                output_path = Path(args.output_path)
                output_path.mkdir(parents=True, exist_ok=True)

                # write the train, val and test filenames to a json file
                for i,image in enumerate(annotation_set["images"]):
                    with open(output_path / f"{data_type}_{image['file_name'].replace('.h5','')}_{set_name}", "w", encoding="utf-8") as f:
                        json.dump([filenames[i]], f)







    else:

        for annotation_set in annotations_sets:
            set_name = Path(annotation_set).name

            # read json file
            with open(annotation_set, "r", encoding="utf-8") as f:
                annotation_set = json.load(f)

            # read the "images" key and for every instance get the "file_name" key
            filenames = [f'{raw_data_path}/{image["file_name"]}' for image in annotation_set["images"]]

            # create a directory to store the folds
            output_path = Path(args.output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            # write the train, val and test filenames to a json file
            with open(output_path / f"{data_type}_{set_name}", "w", encoding="utf-8") as f:
                json.dump(filenames, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations_path", type=Path, default=None, help="Path to the annotations json file.")
    parser.add_argument("output_path", type=Path, default=None, help="Path to the output directory.")
    parser.add_argument("--data_type", choices=["raw", "image"], default="raw", help="Type of data to split.")
    parser.add_argument("--nfold", type=int, default=None, help="Amount of folds to split the train_split")
    args = parser.parse_args()
    main(args)
