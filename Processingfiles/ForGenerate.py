import os
import json
import time
import torch
import logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局模型路径
MODEL_PATH = "/cpfs01/projects-HDD/cfff-0082a359858b_HDD/sxc_22300240012/QwQ-32B"

# 秒数转换为小时-分钟-秒格式
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}小时-{minutes:02d}分钟-{seconds:02d}秒"

# 后处理函数：移除多余的 </answer> 标签
def clean_output_data(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        for entry in data:
            output = entry["output"]
            while output.endswith("</answer></answer>"):
                output = output[:-len("</answer>")]
            entry["output"] = output
        with open(json_file, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"后处理文件 {json_file} 失败: {e}")

# 从回答中提取参考答案
def extract_reference_answer(answer_text):
    start_marker = "答案是："
    start_idx = answer_text.find(start_marker)
    if start_idx == -1:
        return answer_text
    start_idx += len(start_marker)
    end_idx = answer_text.find("。", start_idx)
    if end_idx == -1:
        return answer_text[start_idx:]
    return answer_text[start_idx:end_idx]

# 主处理函数
def process_data(input_data, output_file, start_time):
    # 初始化 vLLM 模型
    try:
        model = LLM(
            model=MODEL_PATH,
            dtype="float16",
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            max_model_len=16384,  # 支持 Prompt + max_tokens=10000
            tensor_parallel_size=torch.cuda.device_count(),  # 自动分配 GPU
            disable_custom_all_reduce=True,
            enforce_eager=True,
            enable_chunked_prefill=False
        )
    except Exception as e:
        logger.error(f"初始化 vLLM 模型失败: {e}")
        return

    # 加载分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        logger.error(f"加载分词器失败: {e}")
        return

    logger.info("模型加载完成，自动分配到可用 GPU")

    # Prompt 模板（简化示例，移除参考答案嵌入）
    PROMPT_TEMPLATE = """你是一位医学专家，请以<think>-[此处为你的思维链]-</think>-<answer>-[此处为你的正式回答]-</answer>的结构回答这个问题：{user_question}{question_mark}。请提供详细的推理过程，确保思维链清晰、逻辑严密，逐步分析问题，探索可能答案，纠正错误，最终得出结论。

这是一个示例，仅作为格式参考，与内容无关：
<think>
嗯，用户问为什么在互联网上人们常常把对手比作比纳粹更坏，甚至让纳粹看起来反而成了好的。这个问题有点复杂，我得先理清楚可能的几个方面。首先，用户可能观察到网络上的极端言论，比如在争论中一方用纳粹来贬低对方，甚至出现“比纳粹更坏”的说法，导致原本作为极端邪恶象征的纳粹反而被相对化了。这可能涉及到几个因素：网络言论的夸张化、历史知识的缺乏、道德比较的滑坡，以及极端化交流的趋势。

接下来，我需要考虑用户为什么会提出这个问题。可能用户本身对这种现象感到困惑，或者担心这种言论对公共讨论的影响。用户可能希望了解背后的社会心理因素，或者这种现象的根源。也有可能用户自己经历过类似的网络争论，想找到解释。

然后，我需要分析可能的原因。比如，互联网的匿名性和快速传播导致人们倾向于使用极端的比喻来吸引注意，而纳粹作为广泛认知的负面象征，被用作攻击工具。但过度使用可能导致“贬值”，反而让更极端的标签出现，形成恶性循环。此外，历史教育的不足可能让一些人并不真正理解纳粹的罪行，从而轻率地进行比较。

还要考虑文化差异，不同地区对纳粹的认知可能不同，比如在某些国家，纳粹的象征意义可能更敏感，而在其他地方，可能被滥用为一般的贬义词。另外，政治极化加剧了这种标签化的攻击，双方都试图将对方妖魔化，导致极端词汇的滥用。

然后，我需要验证这些假设。是否有研究支持网络极端言论的趋势？比如，有心理学研究指出，匿名环境下人们更容易表现出攻击性。同时，历史教育的缺失确实存在，尤其是年轻一代可能对二战历史了解不深。另外，政治极化确实导致更多极端化的标签使用，比如在美国的政治讨论中，左右两派经常互相指责对方为法西斯或纳粹。

可能还需要考虑具体案例，比如在社交媒体上，某些争议话题中，参与者如何用纳粹作比较。例如，在疫苗政策讨论中，反对者可能将支持强制接种的人比作纳粹，而支持者则可能反击对方为反科学、比纳粹更危险。这种情况下，双方的极端化言论可能导致纳粹标签的滥用。

最后，需要总结这些因素，解释为什么会出现这种现象，以及可能的后果，比如削弱对真正极端主义的警惕，或者导致讨论无法理性进行。同时，用户可能希望得到一些解决方案，比如加强历史教育，促进理性对话，或者平台如何管理这类言论。
</think>
<answer>
在互联网上，人们将对手称为“比纳粹更坏”甚至导致“纳粹反而被美化”的现象，是多种复杂社会、心理和传播因素共同作用的结果。以下是这一现象的主要成因和逻辑链条：

一、极端化标签的滥用与历史认知的稀释
简化道德评判的倾向
互联网讨论常追求快速、强烈的情绪表达，而纳粹因其历史共识（系统性种族灭绝、战争罪行）成为“绝对邪恶”的符号。当人们用“纳粹”攻击对手时，本质是将复杂议题简化为“善与恶”的对立，试图通过极端标签直接否定对方立场的合法性。

历史记忆的淡化
随着二战亲历者的逝去，年轻一代对纳粹罪行的认知逐渐抽象化。部分人仅将其视为“坏人的代名词”，而非具体的历史暴行。这种认知真空导致类比被滥用，甚至出现“纳粹不够坏，需要更极端的比喻”的滑坡逻辑。

道德比较的失效
当“纳粹”被频繁用于日常争论（如政策分歧、文化冲突），其象征意义被稀释。为了制造更强的冲击力，一些人开始使用“比纳粹更坏”的表述，形成恶性循环：标签越极端，越需要更极端的标签来“胜出”。

二、互联网传播机制的影响
情绪驱动的算法逻辑
社交媒体平台倾向于推广具有争议性、情绪化的内容。极端言论（如将对手比作纳粹）更容易引发愤怒、恐惧等情绪，从而获得更多曝光，形成“极端言论-流量奖励-更多人效仿”的循环。

匿名性与去责任化
互联网的匿名环境降低了言论的责任成本，使人们更倾向于使用极端比喻攻击对手，而无需顾忌现实中的道德压力或社交后果。

圈层化的信息茧房
在封闭的社群中，极端观点会通过内部重复被不断强化。例如，某些群体可能将外部批评者统称为“新纳粹”，而对立群体则以“反人道”等更极端的标签回击，最终导致双方陷入“比谁更邪恶”的修辞竞赛。

三、社会极化与身份政治的催化
政治对立的武器化
在高度极化的社会环境中（如欧美国家的左右翼冲突），将对手比作纳粹成为一种政治策略。例如，右翼可能指责左翼的“文化压制”类似纳粹的意识形态控制，而左翼则抭击右翼的排外政策“比纳粹更危险”。这种互贴标签的行为加剧了社会的分裂。

身份认同的防御机制
当群体感到自身价值观受到威胁时，倾向于通过妖魔化对手来巩固内部团结。此时，“比纳粹更坏”的指控不仅是攻击，更是强化自身道德优越感的手段。

对“绝对正义”的争夺
在公共议题（如环保、性别平等）中，激进派可能通过极端类比（如“破坏环境=反人类罪”）抢占道德制高点，迫使对手陷入被动辩解，而非理性讨论。

四、后果与反思
消解历史严肃性
纳粹罪行的独特性被模糊，大屠杀等历史记忆可能沦为“网络骂战工具”，削弱公众对真正极端主义的警惕。

阻碍理性对话
极端标签导致讨论焦点从事实和逻辑转向道德审判，使社会陷入“立场优先”的非黑即白思维。

解方建议
强化历史教育：普及纳粹罪行的具体历史细节，避免抽象化、符号化滥用。
平台责任：社交媒体需限制仇恨言论，但需谨慎平衡言论自由与道德边界。
公众媒介素养：鼓励批判性思维，区分“情绪宣泄”与“事实讨论”。

结语
这种现象本质是互联网时代话语权争夺的缩影：当极端化成为吸引注意力的捷径，历史与现实的复杂性便被牺牲。重塑健康的公共讨论，需要从个体认知到平台机制的系统性反思。这道问题的答案是：{reference_answer}
</answer>

请严格按照格式，以<think>标签开头，然后是你进行详细的自我反思、探索和自我纠正的长思维链，以</think>结束思维链，然后以<answer>开始正式回答，以</answer>结束。你要回答的问题是：{user_question}{question_mark}
"""

    # JSON 输出文件
    output_data = []
    batch_size = 5
    batch_data = []
    total_entries = len(input_data)

    # 定义生成回复的函数
    def generate_response(model, prompt):
        sampling_params = SamplingParams(
            max_tokens=10000,
            temperature=0.5,  # 降低 temperature 增加稳定性
            top_p=0.9
        )
        try:
            outputs = model.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text
            return response
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            return ""

    for i, entry in enumerate(input_data):
        user_question = entry["conversations"][0]["value"]
        reference_answer_raw = entry["conversations"][1]["value"]
        reference_answer = extract_reference_answer(reference_answer_raw)

        question_mark = "" if user_question.endswith(('?', '？')) else "？"
        prompt = PROMPT_TEMPLATE.format(user_question=user_question, question_mark=question_mark, reference_answer=reference_answer)

        # 调试：记录 Prompt 长度
        logger.debug(f"条目 {i+1}/{total_entries}，问题: {user_question}")
        logger.debug(f"Prompt 长度: {len(prompt)}")

        # 生成模型回复
        model_response = generate_response(model, prompt)

        # 调试：记录 model_response 内容和长度
        logger.debug(f"model_response 长度: {len(model_response)}")
        logger.debug(f"model_response 内容: {model_response[:200]}...")  # 截取前 200 字符

        # 剪裁 Prompt，保留模型实际生成的回答
        extracted_response = model_response[len(prompt):] if len(model_response) > len(prompt) else model_response

        # 调试：记录 extracted_response 内容和长度
        logger.debug(f"extracted_response 长度: {len(extracted_response)}")
        logger.debug(f"extracted_response 内容: {extracted_response[:200]}...")

        # 直接使用剪裁后的生成内容，不进行格式判定或包装
        json_entry = {
            "instruction": prompt,
            "input": user_question,
            "output": extracted_response
        }
        batch_data.append(json_entry)

        # 定期写入缓冲区到文件
        if len(batch_data) >= batch_size or i == total_entries - 1:
            output_data.extend(batch_data)
            try:
                with open(output_file, 'w', encoding='utf-8') as outfile:
                    json.dump(output_data, outfile, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"写入文件 {output_file} 失败: {e}")
            batch_data = []

        # 计算进度和剩余时间
        elapsed_time = time.time() - start_time
        processed_count = i + 1
        avg_time_per_entry = elapsed_time / processed_count if processed_count > 0 else 0
        remaining_entries = total_entries - processed_count
        estimated_remaining_time = avg_time_per_entry * remaining_entries
        file_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
        print(f"已处理 {processed_count}/{total_entries} 条，文件大小：{file_size} 字节，"
              f"已用时间：{format_time(elapsed_time)}，预计剩余时间：{format_time(estimated_remaining_time)}", flush=True)

    # 后处理文件
    clean_output_data(output_file)

# 主函数
def main():
    # 计时器开始
    start_time = time.time()

    # 定义输入和输出文件路径
    input_json_file = "/cpfs01/projects-HDD/cfff-0082a359858b_HDD/sxc_22300240012/Qwen-main/Qwen-main/data/common_medical_train.json"
    output_json_file = "common_medical_qwq32b_with_example_nazi_prompt_and_answer.json"

    # 最大处理条数
    max_entries_to_process = 3000

    # 读取 JSON 文件
    try:
        with open(input_json_file, 'r', encoding='utf-8') as input_file:
            input_data = json.load(input_file)
            total_entries = len(input_data)
            if max_entries_to_process is not None and max_entries_to_process < total_entries:
                input_data = input_data[:max_entries_to_process]
                total_entries = max_entries_to_process
            print(f"总数据条数: {total_entries}")
    except FileNotFoundError:
        print(f"错误：输入文件 {input_json_file} 不存在！")
        logger.error(f"输入文件 {input_json_file} 不存在")
        return
    except json.JSONDecodeError:
        print(f"错误：输入文件 {input_json_file} 格式不正确！")
        logger.error(f"输入文件 {input_json_file} 格式不正确")
        return

    # 处理数据
    process_data(input_data, output_json_file, start_time)

    # 计时器结束并输出总时间
    total_time = time.time() - start_time
    print(f"\n处理完成，结果保存到 {output_json_file}")
    print(f"总耗时: {format_time(total_time)}")

if __name__ == "__main__":
    main()