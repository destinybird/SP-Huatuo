# SP-Huatuo: A Self-Play Enhanced Medical Reasoning Model

## English README

### Introduction
SP-Huatuo is a 7B-parameter small language model designed for advanced medical reasoning, built upon the Qwen2.5-7B-Instruct base model. The name "SP" stands for **Self-Play**, a novel two-stage training framework that enhances reasoning capabilities through self-collaborative correction, as detailed in the paper *Self-Play: Enhancing Medical Reasoning in Small LLMs via Self-Collaborative Correction*. Additionally, "SP" pays homage to the prefix used for new characters in the game *Three Kingdoms Kill*, celebrating the legacy of the legendary medical model **Huatuo-o1-7B**. SP-Huatuo achieves state-of-the-art (SOTA) performance among 7B-class models on the CMExam benchmark, surpassing strong baselines like Taiyi and Huatuo-o1-7B.

The model files are hosted on ModelScope at [https://www.modelscope.cn/models/suxinchun/Self-Play/summary](https://www.modelscope.cn/models/suxinchun/Self-Play/summary). This repository contains supporting files for training, processing, and testing the SP-Huatuo model, including loss visualizations, processing scripts, and training documents.

### Key Features
- **Self-Play Framework**: Utilizes a two-stage approach:
  1. **Supervised Fine-Tuning (SFT)** with long chain-of-thought templates derived from a teacher model (QwQ-32B).
  2. **Model Self-Collaborative Direct Preference Optimization (DPO)** to enhance reasoning through self-correction.
- **Superior Performance**: Achieves **70.62% accuracy** on the CMExam dataset, outperforming Taiyi by **4.16%** and Huatuo-o1-7B by **11.24%**.
- **Medical Expertise**: Tailored for complex medical reasoning tasks, leveraging high-quality datasets with 13,000 medical question-answering samples.
- **Efficient and Cost-Effective**: Optimizes small LLMs to rival larger models without the high computational costs of teacher-guided methods.

### Repository Structure
This repository contains the following folders and files:

- **LossFigures**: Contains visualizations of the loss curves during training, showcasing the model's convergence and performance improvements.
- **ProcessingFiles**: Includes three Python scripts for data processing:
  - `ForGenerate.py`: Used by the teacher model (QwQ-32B) to generate reasoning chain corpora for fine-tuning.
  - `ForJudge.py`: Enables the teacher model to evaluate the correctness of student model outputs (binary judgment: correct or incorrect).
  - `ForTest.py`: Facilitates testing of the student model (SP-Huatuo) on the CMExam dataset.
- **TrainingDocuments**: Contains training configuration files and datasets used with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for fine-tuning and DPO optimization.

All test results mentioned in the paper are available in the [Releases section](https://github.com/destinybird/SP-Huatuo/releases) of this repository.

### Model Details
- **Base Model**: Qwen2.5-7B-Instruct
- **Training Dataset**: 13,000 medical Q&A samples, split into 3,000 for thought-level error correction and 10,000 for testing.
- **Teacher Model**: QwQ-32B for generating reasoning chains.
- **Framework**: Self-Play, combining long chain-of-thought SFT and self-collaborative DPO.
- **Training Framework**: LLaMA-Factory
- **License**: Apache License 2.0
- **Model Hosting**: Available on ModelScope at [https://www.modelscope.cn/models/suxinchun/Self-Play/summary](https://www.modelscope.cn/models/suxinchun/Self-Play/summary).

### Usage
To use SP-Huatuo, download the model from ModelScope and follow these steps:

1. **Installation**:
   ```bash
   pip install modelscope transformers
   ```

2. **Loading the Model**:
   ```python
   from modelscope.hub.snapshot_download import snapshot_download
   from transformers import AutoModelForCausalLM, AutoTokenizer

   # Download model
   model_dir = snapshot_download('suxinchun/Self-Play')
   
   # Load model and tokenizer
   tokenizer = AutoTokenizer.from_pretrained(model_dir)
   model = AutoModelForCausalLM.from_pretrained(model_dir)
   
   # Example inference
   prompt = "What are the chest pain characteristics of dry pleurisy?"
   inputs = tokenizer(prompt, return_tensors="pt")
   outputs = model.generate(**inputs, max_length=500)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

3. **Inference Tips**:
   - Use structured prompts with `<think>` and `<answer>` tags for optimal reasoning output.
   - Adjust `max_length` based on the complexity of the medical query.

To replicate training or testing, refer to the scripts in `ProcessingFiles` and the configurations in `TrainingDocuments`. Use LLaMA-Factory for training, following the instructions in its [documentation](https://github.com/hiyouga/LLaMA-Factory).

### Performance
SP-Huatuo sets a new SOTA for 7B models on the CMExam dataset:
- **Accuracy**: 70.62% (Total Set)
- **Comparison**:
  - Taiyi: 66.46% (+4.16%)
  - Huatuo-o1-7B: 59.38% (+11.24%)
  - DeepSeek-R1-Distill-Qwen-7B: 57.47% (+13.15%)

Test results and detailed ablation studies are available in the [Releases section](https://github.com/destinybird/SP-Huatuo/releases).

### Disclaimer
SP-Huatuo is a research model intended for academic and experimental use only. It is not a substitute for professional medical advice, diagnosis, or treatment. Users should consult qualified healthcare professionals for medical decisions. The developers are not liable for any misuse or consequences arising from the model's outputs.

### Acknowledgments
- **Research Funding**: Supported by the FDUROP program of Fudan University (No. 24254).
- **Computing Platform**: CFFF computing platform.
- **Legacy Inspiration**: Huatuo-o1-7B, a pioneering medical model.

### Contact
For inquiries, contact: Xinchun Su (22300240012@m.fudan.edu.cn).

---

## 中文版 README

### SP-华佗：基于自博弈的医疗推理增强模型

#### 简介
SP-华佗是一个7B参数的小型语言模型，专为高级医疗推理设计，基于Qwen2.5-7B-Instruct基础模型构建。名称“SP”代表**Self-Play**（自博弈），这是一种新颖的两阶段训练框架，通过自协作纠错提升推理能力，详见论文 *Self-Play: Enhancing Medical Reasoning in Small LLMs via Self-Collaborative Correction*。此外，“SP”还致敬了《三国杀》游戏中新武将的常见前缀，向传奇医疗模型**Huatuo-o1-7B**致敬。SP-华佗在CMExam基准测试中实现了7B级别模型的SOTA性能，超越了Taiyi和Huatuo-o1-7B等强大基线。

模型文件托管在ModelScope上，地址为：[https://www.modelscope.cn/models/suxinchun/Self-Play/summary](https://www.modelscope.cn/models/suxinchun/Self-Play/summary)。本仓库包含训练、处理和测试SP-华佗模型的支持文件，包括损失可视化、处理脚本和训练文档。

#### 主要特性
- **自博弈框架**：采用两阶段方法：
  1. 使用从教师模型（QwQ-32B）提取的长链思维模板进行**监督微调（SFT）**。
  2. 通过**模型自协作直接偏好优化（DPO）**，通过自纠错增强推理能力。
- **卓越性能**：在CMExam数据集上实现**70.62%的准确率**，比Taiyi高**4.16%**，比Huatuo-o1-7B高**11.24%**。
- **医疗专业性**：专为复杂医疗推理任务定制，利用包含13,000个医疗问答样本的高质量数据集。
- **高效且成本低**：优化小型LLM，使其在无需高计算成本的教师指导方法下与大型模型媲美。

#### 仓库结构
本仓库包含以下文件夹和文件：

- **LossFigures**：包含训练过程中的损失曲线可视化，展示模型的收敛性和性能提升。
- **ProcessingFiles**：包含三个用于数据处理的Python脚本：
  - `ForGenerate.py`：教师模型（QwQ-32B）用于生成推理链语料以进行微调。
  - `ForJudge.py`：教师模型用于评估学生模型输出的正确性（二元判断：正确或错误）。
  - `ForTest.py`：用于在CMExam数据集上测试学生模型（SP-华佗）。
- **TrainingDocuments**：包含使用[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)进行微调和DPO优化的训练配置文件和数据集。

论文中提到的所有测试结果可在仓库的[Releases页面](https://github.com/destinybird/SP-Huatuo/releases)中找到。

#### 模型详情
- **基础模型**：Qwen2.5-7B-Instruct
- **训练数据集**：13,000个医疗问答样本，分为3,000个用于思维级纠错，10,000个用于测试。
- **教师模型**：QwQ-32B，用于生成推理链。
- **框架**：自博弈，结合长链思维SFT和自协作DPO。
- **训练框架**：LLaMA-Factory
- **许可证**：Apache License 2.0
- **模型托管**：ModelScope，地址为 [https://www.modelscope.cn/models/suxinchun/Self-Play/summary](https://www.modelscope.cn/models/suxinchun/Self-Play/summary)。

#### 使用方法
要使用SP-华佗，请从ModelScope下载模型并按照以下步骤操作：

1. **安装**：
   ```bash
   pip install modelscope transformers
   ```

2. **加载模型**：
   ```python
   from modelscope.hub.snapshot_download import snapshot_download
   from transformers import AutoModelForCausalLM, AutoTokenizer

   # 下载模型
   model_dir = snapshot_download('suxinchun/Self-Play')
   
   # 加载模型和分词器
   tokenizer = AutoTokenizer.from_pretrained(model_dir)
   model = AutoModelForCausalLM.from_pretrained(model_dir)
   
   # 示例推理
   prompt = "干性胸膜炎的胸痛特征是什么？"
   inputs = tokenizer(prompt, return_tensors="pt")
   outputs = model.generate(**inputs, max_length=500)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

3. **推理提示**：
   - 使用带有`<think>`和`<answer>`标签的结构化提示，以获得最佳推理输出。
   - 根据医疗查询的复杂性调整`max_length`。

要重现训练或测试，请参考`ProcessingFiles`中的脚本和`TrainingDocuments`中的配置。使用LLaMA-Factory进行训练，遵循其[文档](https://github.com/hiyouga/LLaMA-Factory)中的说明。

#### 性能
SP-华佗在CMExam数据集上为7B模型设定了新的SOTA：
- **准确率**：70.62%（总数据集）
- **对比**：
  - Taiyi：66.46%（+4.16%）
  - Huatuo-o1-7B：59.38%（+11.24%）
  - DeepSeek-R1-Distill-Qwen-7B：57.47%（+13.15%）

测试结果和详细的消融研究可在[Releases页面](https://github.com/destinybird/SP-Huatuo/releases)中找到。

#### 免责声明
SP-华佗仅为学术和实验用途的研究模型，不得用于替代专业医疗建议、诊断或治疗。用户应咨询合格的医疗专业人员以做出医疗决定。开发者对模型输出的误用或由此产生的后果不承担责任。

#### 致谢
- **研究资助**：复旦大学FDUROP计划（编号24254）。
- **计算平台**：CFFF计算平台。
- **历史传承**：Huatuo-o1-7B，医疗模型的先驱。

#### 联系方式
如有疑问，请联系：Xinchun Su (22300240012@m.fudan.edu.cn)。