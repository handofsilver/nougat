import json
import os
import shutil
from create_data import walk_and_create


data_json_file = "/home/ninziwei/lyj/nougat/nougat/gen_dataset/data.json"
dataset_root = "/data1/nzw/latex_pdf/generated_dataset"


def process_zip_file(zip_file):
    try:
        if walk_and_create(zip_file, dataset_root):
            with open("success.txt", "a") as f:
                f.write(zip_file + "\n")
            print(f"Create Data Done: {zip_file}")
    except Exception as e:
        print(f"Create Data Failed: {e}")


def main():
    with open(data_json_file, "r") as f:
        data = json.load(f)
        zip_files = data["zip_files"]

    for zip_file in zip_files:
        pdf_file = zip_file.replace(".zip", ".pdf")

        try:
            target_zip_file = os.path.join(
                dataset_root, "src", os.path.basename(zip_file)
            )
            target_pdf_file = os.path.join(
                dataset_root, "src", os.path.basename(pdf_file)
            )

            if not os.path.exists(target_zip_file):
                shutil.copy(zip_file, target_zip_file)
                shutil.copy(pdf_file, target_pdf_file)

                print(f"Copy {zip_file} to {target_zip_file} done")
                process_zip_file(target_zip_file)
        except Exception as e:
            print(f"Error processing {zip_file}: {e}")


if __name__ == "__main__":
    main()
