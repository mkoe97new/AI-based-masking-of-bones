import zipfile
from pathlib import Path
import argparse
import datetime
from tqdm import tqdm


def zip_folder(folder_path, output_name=None):
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"{folder_path} is not a valid folder")

    if output_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{folder.name}_{timestamp}.zip"

    zip_path = folder.parent / output_name

    files = [f for f in folder.rglob("*") if f.is_file()]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for file in tqdm(files, desc="Zipping", unit="file"):
            z.write(file, arcname=file.relative_to(folder))

    return zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zip a folder")
    parser.add_argument(
        "--zip_folder",
        type=str,
        required=True,
        help="Path to the folder to zip"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional name of the output zip file"
    )

    args = parser.parse_args()

    zip_path = zip_folder(args.zip_folder, args.output_name)
    print(f"Created zip: {zip_path}")