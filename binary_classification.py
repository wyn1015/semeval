from modelscope import AutoModelForCausalLM, AutoTokenizer
import json

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line.strip())
            data.append(json_data)
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

# 指定本地模型路径
local_model_path = ""

# 从本地路径加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 读取 JSONL 文件
data = load_jsonl('')

# 准备保存结果的数据列表
results = []

# 构建并处理每条消息
for item in data:
    compound = item['compound'][0]  # 假设compound是一个列表，取第一个元素
    text = item['text']
    
    # 构建提示信息
    prompt = "Please analyze the use of the {compound} word in the given {text} text to determine whether the word is idiomatic or literal . For instance, dark is used metaphorically in the sentence 'The latest developments move us closer to a dark age', but not in 'I like dark colors'. Just output \"idiomatic\" or \"literal\"."

    messages = [
        # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant in Portuguese."},
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant in English."},
        {"role": "user", "content": prompt.format(compound=compound, text=text)}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 如果原先有answer字段，将其更名为answer1
    if 'answer' in item:
        item['answer1'] = item.pop('answer')
    
    # 将模型推理出的response作为新的answer字段
    item['answer'] = response

    # 将更新后的条目保存到列表中
    results.append(item)

# 保存结果到新的 JSONL 文件
save_jsonl(results, '')