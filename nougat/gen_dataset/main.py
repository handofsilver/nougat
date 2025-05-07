import json
import subprocess
import threading
from select_files import is_zip_wanted, find_all_zips
from create_data import walk_and_create
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import psutil  # 需要安装: pip install psutil
from functools import partial


latex_pdf_root = "/data1/nzw/latex_pdf/"
target_json_file = "/home/ninziwei/lyj/nougat/gen_dataset/data.json"
required_size = 10000  # The number of files to generate
generated_size = 0
# 添加锁以保护全局变量
size_lock = threading.Lock()
# 添加一个事件标志来指示是否已完成工作
done_event = threading.Event()


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


def process_zip_file(zip_file):
    global generated_size

    # 如果已完成，直接返回
    if done_event.is_set():
        return False

    try:
        if walk_and_create(zip_file):
            # 使用锁保护对共享变量的修改
            with size_lock:
                generated_size += 1
                current_size = generated_size  # 在锁内获取当前值

                with open("success.txt", "a") as f:
                    f.write(zip_file + "\n")

                print(f"create data: {current_size}/{required_size}")

                # 检查是否达到目标
                if current_size >= required_size:
                    done_event.set()  # 设置事件标志
                    return True

    except Exception as e:
        print(f"create data failed: {e}")
    return False


def process_zip_file_with_retry(zip_file, max_retries=3):
    """带重试机制的处理函数"""
    retries = 0
    while retries < max_retries:
        try:
            return process_zip_file(zip_file)
        except Exception as e:
            retries += 1
            if retries < max_retries:
                print(f"处理{zip_file}失败: {e}，尝试第{retries}次重试")
                time.sleep(1)  # 重试前等待一段时间
            else:
                print(f"处理{zip_file}失败，已达到最大重试次数: {e}")
                return False


def main():
    # process zip files
    with open(target_json_file, "r") as f:
        data = json.load(f)
        zip_files = data["zip_files"]

    batch_size = 100

    start_time = time.time()
    processed_count = 0

    # 根据CPU核心数动态设置线程数
    cpu_count = psutil.cpu_count(logical=False)  # 物理核心数
    max_workers = max(1, min(cpu_count - 1, 10))  # 保留至少一个核心给系统

    print(f"使用{max_workers}个工作线程 (系统有{cpu_count}个CPU核心)")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(zip_files), batch_size):
            batch_start = time.time()

            subprocess.run("rm *.log", shell=True)

            if generated_size >= required_size or done_event.is_set():
                break

            # 根据CPU使用率动态调整并发任务数
            cpu_usage = psutil.cpu_percent(interval=0.5)

            # 如果CPU使用率过高，减少此批次的任务数
            if cpu_usage > 90:
                current_batch_size = batch_size // 2
                print(f"CPU使用率高({cpu_usage}%)，减少批次大小至{current_batch_size}")
            else:
                current_batch_size = batch_size

            batch = zip_files[i : i + current_batch_size]

            # 使用带重试的版本
            futures = [
                executor.submit(process_zip_file_with_retry, zip_file)
                for zip_file in batch
            ]

            for future in as_completed(futures):
                try:
                    if future.result():
                        # 如果达到目标，取消所有未完成的任务
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
                except Exception as e:
                    print(f"任务执行出错: {e}")

                # 检查是否已达到目标
                if done_event.is_set():
                    break

            # 如果已达到目标，跳出循环
            if done_event.is_set():
                print("已达到目标数量，终止处理")
                break

            processed_count += len(batch)
            batch_time = time.time() - batch_start
            elapsed = time.time() - start_time

            # 打印详细的性能统计
            print(f"批次 {i//batch_size + 1} 完成:")
            print(f"  - 已处理: {processed_count}个文件")
            print(f"  - 已生成: {generated_size}/{required_size}")
            print(f"  - 批次耗时: {batch_time:.2f}秒")
            print(f"  - 总耗时: {elapsed:.2f}秒")
            print(f"  - 处理速度: {generated_size/elapsed:.2f}个/秒")
            print(f"  - CPU使用率: {psutil.cpu_percent()}%")
            print(f"  - 内存使用: {psutil.virtual_memory().percent}%")

    print("create data done!")


if __name__ == "__main__":
    main()
