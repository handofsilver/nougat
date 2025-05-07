import json
from select_files import is_zip_wanted, find_all_zips


latex_pdf_root = "/data1/nzw/latex_pdf/"
target_json_file = "/home/ninziwei/lyj/nougat/nougat/gen_dataset/data.json"
required_size = 10000  # The number of files to generate
generated_size = 0


def write_json(zip_files, size=10):
    zips = []
    for zip_file in zip_files:
        try:
            if is_zip_wanted(zip_file):
                zips.append(zip_file)
                print(f"select files: {len(zips)}/{size}")
                if len(zips) >= size:
                    break
        except Exception as e:
            print(f"select files failed: {e}")
            continue

    data = {"zip_files": zips}
    with open(target_json_file, "w") as f:
        json.dump(data, f)


def main():
    # select zip files and write to json
    zip_files = find_all_zips(latex_pdf_root)
    write_json(zip_files, size=required_size)
    print("select files done!")
    pass


if __name__ == "__main__":
    main()
