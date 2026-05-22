import kagglehub
from pathlib import Path
import random
import shutil
import os


def download_dataset():
    # Download latest version
    path = kagglehub.dataset_download("rajeevpaudel1/nepali-food-image-dataset")

    print("Path to dataset files:", path)
def download_verification():

    for split in ["train", "test"]:

        print(f"\n{split.upper()}")

        split_path = Path("dataset") / split

        for cls in split_path.iterdir():

            count = len(list(cls.glob("*")))

            print(cls.name, count)
def split_data():
    random.seed(42)
    source_dir = Path("nepali-food-image-dataset/versions/1/dataset/")
    train_dir = Path("dataset/train")
    test_dir = Path("dataset/test")

    # walk through raw image dir
    # for dirpath, dirnames, filenames in os.walk(source_dir):
    #     print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    # make train anand test Dir
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    classes = [folder for folder in source_dir.iterdir() if folder.is_dir()]
    # print(classes)
    for cls in classes:
        # print(cls)
        image = list(cls.glob("images/*"))
        # print(image[:5])
        random.shuffle(image)
        # split Index
        split_idx = int(0.8 * len(image))

        train_image = image[:split_idx]
        test_image = image[split_idx:]

        # create destination folders
        (train_dir / cls.name).mkdir(parents=True, exist_ok=True)
        (test_dir / cls.name).mkdir(parents=True, exist_ok=True)
        # copy train images
        for img in train_image:
            shutil.copy(src=img,dst=train_dir/cls.name/img.name)
        for img in test_image:
            shutil.copy(src=img,dst=test_dir/cls.name/img.name)
        print(f"{cls.name}:")
        print(f"  Train: {len(train_image)}")
        print(f"  Test : {len(test_image)}")
        

    print("\nDataset split complete.")
download_verification()