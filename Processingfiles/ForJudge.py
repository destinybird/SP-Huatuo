import csv
import os
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 秒数转换为小时-分钟-秒格式
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}小时:{minutes:02d}分钟:{seconds:02d}秒"

# 计时器开始
start_time = time.time()

# 指定 QwQ-32B 模型路径
model_path = "/cpfs01/projects-HDD/cfff-0082a359858b_HDD/sxc_22300240012/QwQ-32B"

# 加载 vLLM 模型
try:
    llm = LLM(
        model=model_path,
        dtype="auto",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,  # 使用 2 个 GPU
        max_model_len=4096,
        disable_custom_all_reduce=True,
        enforce_eager=True,
        enable_chunked_prefill=False
    )
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)

# 加载分词器
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
except Exception as e:
    print(f"分词器加载失败: {e}")
    exit(1)

# 修复 Attention Mask 警告：设置 pad_token_id
if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

# 读取输入 CSV 文件路径
csv_input_file = "/cpfs01/projects-HDD/cfff-0082a359858b_HDD/sxc_22300240012/LLaMA-Factory/test_result_common_medical_for_test_total_10887_qwen2_7b_common_medical_nazi_jingjian_cot_sft_sixuan_dpo.csv"

# 定义处理的行数
num_rows_to_process = 10887

# 动态生成输出 CSV 文件路径
csv_output_file = f"test_result_judged_{num_rows_to_process}_{os.path.basename(csv_input_file)}"

# 打开 CSV 文件，读取问题和小模型的回复
try:
    with open(csv_input_file, 'r', encoding='utf-8', errors='ignore') as infile, \
         open(csv_output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 写入表头
        header = next(reader, None)  # 跳过表头
        if not header:
            print("错误：CSV 文件为空或无效")
            exit(1)
        writer.writerow(header + ['判断结果'])  # 添加新列

        # 初始化计数器
        total_processed = 0
        correct_count = 0
        incorrect_count = 0

        # 获取 1 和 2 的 token ID
        allowed_token_ids = [
            tokenizer.convert_tokens_to_ids("1"),
            tokenizer.convert_tokens_to_ids("2")
        ]

        # 遍历文件中的每一行
        for row in reader:
            if total_processed >= num_rows_to_process:
                break

            try:
                user_question = row[0]
                answer = row[1]
                small_model_response_full = row[2]

                # 提取小模型回复
                if "只解释这一个问题即可" in small_model_response_full:
                    small_model_response = small_model_response_full.split("只解释这一个问题即可")[-1].strip()
                else:
                    small_model_response = small_model_response_full.strip()

                # 使用原始提示
                prompt = (
                    f"这是一位小模型对问题的回复内容，根据其内容判断其是否做到了正确回复。问题：{user_question}？"
                    f"答案是：{answer}，小模型回复：{small_model_response}。"
                    f"如果正确，则只直接回复1。如果错误，则只直接回复2"
                )

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]

                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # 配置生成参数
                sampling_params = SamplingParams(
                    max_tokens=1,
                    temperature=0.01,  # 极低温度，确保确定性
                    top_k=2,  # 限制 top-2 token
                    allowed_token_ids=allowed_token_ids  # 强制只生成 1 或 2
                )

                # 生成
                outputs = llm.generate([text], sampling_params)
                response = outputs[0].outputs[0].text.strip()

                total_processed += 1
                if response == '1':
                    correct_count += 1
                elif response == '2':
                    incorrect_count += 1

                # 计算正确率
                accuracy_one = correct_count / total_processed if total_processed > 0 else 0
                accuracy_two = 1 - (incorrect_count / total_processed) if total_processed > 0 else 0

                # 写入结果
                row.append(response)
                writer.writerow(row)
                outfile.flush()

                # 监控文件大小
                file_size = os.path.getsize(csv_output_file)

                # 打印进度
                print(f"已处理 {total_processed} 条，判断结果：{response}，"
                      f"CSV 文件大小：{file_size} 字节，当前正确率1：{accuracy_one:.2%}，"
                      f"当前正确率2：{accuracy_two:.2%}", flush=True)

            except Exception as e:
                print(f"跳过错误行: {row}，错误信息：{e}", flush=True)
                continue

except FileNotFoundError:
    print(f"错误：输入 CSV 文件 {csv_input_file} 不存在")
    exit(1)
except Exception as e:
    print(f"处理 CSV 时出错: {e}")
    exit(1)

# 计时器结束并输出总时间
end_time = time.time()
total_time = end_time - start_time
print(f"\n所有数据处理完毕，结果已保存到 {csv_output_file}")
print(f"总耗时: {total_time:.2f} 秒")