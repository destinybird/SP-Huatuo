import os
import json
import time
import csv
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from multiprocessing import Process, Manager

# 全局模型路径
MODEL_PATH = os.getenv('MODEL_PATH', "/cpfs01/projects-HDD/cfff-0082a359858b_HDD/sxc_22300240012/LLaMA-Factory/output/qwen2_7b_common_medical_nazi_jingjian_cot_sft_sixuan_dpo")

# 秒数转换为小时-分钟-秒格式
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}小时-{minutes:02d}分钟-{seconds:02d}秒"

# 单个进程的处理函数
def process_entries(start_idx, end_idx, gpu_id, input_data, csv_output_file, shared_dict):
    # 计时器开始
    process_start_time = time.time()

    # 设置环境变量以绑定 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 初始化 vLLM 模型
    model = LLM(
        model=MODEL_PATH,
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,  # 单 GPU 模式
        disable_custom_all_reduce=True,
        enforce_eager=True,
        enable_chunked_prefill=False
    )

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 临时 CSV 文件
    temp_csv_file = f"{csv_output_file}_gpu{gpu_id}_temp.csv"

    # 初始化临时 CSV 文件并写入表头
    with open(temp_csv_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["问题", "答案", "小模型回复"])

    buffer = []
    buffer_size = 100

    # 定义生成回复的函数
    def generate_response(model, prompt):
        sampling_params = SamplingParams(
            max_tokens=450,
            temperature=0.7,
            top_p=0.9
        )
        outputs = model.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text
        return response

    # 处理指定范围的数据
    with open(temp_csv_file, 'a', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)

        for i in range(start_idx, end_idx):
            entry = input_data[i]
            user_question = entry['conversations'][0]['value']
            assistant_answer = entry['conversations'][1]['value']

            # 提取答案部分
            if "答案是：" in assistant_answer and "这是一些相关的背景信息" in assistant_answer:
                extracted_answer = assistant_answer.split("答案是：")[1].split("这是一些相关的背景信息")[0].strip()
            else:
                extracted_answer = assistant_answer

            # 构建 Prompt
            prompt = f"你是一位医学专家，请先深呼吸，然后用中文一步一步地思考并回答这个问题，给出分点罗列的解释：问题：{user_question}？只解释这一个问题即可"
            # prompt = f"{user_question}？"

            # 生成模型回复
            model_response = generate_response(model, prompt)

            # 将数据加入缓冲区
            buffer.append([user_question, extracted_answer.strip(), model_response.strip()])

            # 定期写入缓冲区到临时文件
            if len(buffer) >= buffer_size or i == end_idx - 1:
                writer.writerows(buffer)
                outfile.flush()
                buffer.clear()

            # 计算进度和剩余时间
            processed_count = i - start_idx + 1
            total_in_process = end_idx - start_idx
            elapsed_time = time.time() - process_start_time
            avg_time_per_entry = elapsed_time / processed_count if processed_count > 0 else 0
            remaining_entries = total_in_process - processed_count
            estimated_remaining_time = avg_time_per_entry * remaining_entries

            # 打印进度
            print(f"GPU {gpu_id} 已处理 {processed_count}/{total_in_process} 条，临时文件大小：{os.path.getsize(temp_csv_file)} 字节，已用时间：{format_time(elapsed_time)}，预计剩余时间：{format_time(estimated_remaining_time)}", flush=True)

    # 更新共享字典，表示该进程完成
    shared_dict[f"gpu{gpu_id}_done"] = True

# 主函数
def main():
    # 计时器开始
    start_time = time.time()

    # 定义输入和输出文件路径
    json_input_file = "/cpfs01/projects-HDD/cfff-0082a359858b_HDD/sxc_22300240012/Qwen-main/Qwen-main/data/common_medical_test_total.json"
    model_name = os.path.basename(MODEL_PATH)
    num_rows_to_process = 10887
    csv_output_file = f"test_result_common_medical_for_test_total_{num_rows_to_process}_{model_name}.csv"

    # 确保输出目录存在
    output_dir = os.path.dirname(csv_output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取 JSON 文件
    try:
        with open(json_input_file, 'r', encoding='utf-8') as jsonfile:
            input_data = json.load(jsonfile)
            print(f"加载的数据条数: {len(input_data)}")
    except FileNotFoundError:
        print(f"错误：输入文件 {json_input_file} 不存在！")
        return
    except json.JSONDecodeError:
        print(f"错误：输入文件 {json_input_file} 格式不正确！")
        return

    # 分配任务：两块 GPU 各处理一半
    mid_point = num_rows_to_process // 2
    ranges = [(0, mid_point, 0), (mid_point, num_rows_to_process, 1)]  # (start, end, gpu_id)

    # 使用 Manager 共享进程状态
    manager = Manager()
    shared_dict = manager.dict()

    # 启动两个进程
    processes = []
    for start_idx, end_idx, gpu_id in ranges:
        shared_dict[f"gpu{gpu_id}_done"] = False
        p = Process(target=process_entries, args=(start_idx, end_idx, gpu_id, input_data, csv_output_file, shared_dict))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    while not all(shared_dict.values()):
        time.sleep(1)

    for p in processes:
        p.join()

    # 合并临时 CSV 文件
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["问题", "答案", "小模型回复"])  # 写入表头

        for _, _, gpu_id in ranges:
            temp_file = f"{csv_output_file}_gpu{gpu_id}_temp.csv"
            with open(temp_file, 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                next(reader)  # 跳过表头
                for row in reader:
                    writer.writerow(row)
            os.remove(temp_file)  # 删除临时文件

    # 计时器结束并输出总时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n所有数据的处理结果已保存到 CSV 文件：{csv_output_file}")
    print(f"总耗时: {format_time(total_time)}")

if __name__ == "__main__":
    main()