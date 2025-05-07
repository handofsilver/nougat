import os
import re
import shutil
import random
from unzip import extract_files_from_tar_zip


def contains_documentclass(tex_file_path):
    """检查文件是否包含 \documentclass{} 或 \documentclass[]{} 标签"""
    pattern = re.compile(r"\\documentclass(\[[^\]]*\])?\{[^}]*\}")
    with open(tex_file_path, "r", encoding="utf-8", errors="ignore") as file:
        tex_content = file.read()
    if pattern.search(tex_content):
        return True
    return False


def is_zip_wanted(zip_path):
    pdf_path = zip_path.replace(".zip", ".pdf")
    if not os.path.exists(pdf_path):
        return False

    target_dir = zip_path.replace(".zip", "")
    if os.path.exists(target_dir):
        for root, _, files in os.walk(target_dir):
            for file in files:
                # found files like main-*.tex
                if file.endswith(".tex") and file.startswith("main-"):
                    # shutil.rmtree(target_dir)
                    return False

    extract_files_from_tar_zip(zip_path)

    # check if there is only one .tex file in the directory
    # and it contains \documentclass{}
    for root, _, files in os.walk(target_dir):
        tex_count, only_tex_file = 0, None
        for file in files:
            if file.endswith(".tex"):
                tex_count += 1
                only_tex_file = os.path.join(root, file)

        if tex_count != 1:
            shutil.rmtree(target_dir)
            return False
        elif tex_count == 1 and not contains_documentclass(only_tex_file):
            shutil.rmtree(target_dir)
            return False

    # check if there is a .pdf file(PDF-FORMED FIGURE!) in the directory
    for root, _, files in os.walk(target_dir):
        for file in files:
            # no pdf-form figures permitted!
            if file.endswith(".pdf"):
                shutil.rmtree(target_dir)
                return False
    shutil.rmtree(target_dir)

    return True


def find_all_zips(latex_pdf_root):
    zip_files = []

    for root, _, files in os.walk(latex_pdf_root):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".zip"):
                zip_files.append(file_path)

    random.shuffle(zip_files)
    return zip_files
