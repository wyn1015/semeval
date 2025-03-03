import base64
import json
from argparse import ArgumentParser
from tqdm import tqdm
import os
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="",  # 替换为你的 OpenAI API 密钥
    base_url="https://api-inference.modelscope.cn/v1/"  # OpenAI API 的默认地址
)
#示例，具体得根据语言不同，进行修改
# BASE_TEXT_PROMPTS = """You will get a text {text} that contains a specific compound word {compound} and a series of images. You must output the image which best matches the meaning of {compound} in the {text}，not compound itself.
# ###Output
# """
# BASE_TEXT_PROMPTS2 = """You will get a text {text} that contains a specific compound word {compound} and a series of images. You must output the image which best matches the meaning of {compound} in the {text}.The meaning of compound  in the text is fully idiomatic or literal rather than partially idiomatic or literal.
# ###Output
# """
def image_to_base64(image_path):
    """将图片文件转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_message(images=None, text=None, compound=None):
    messages = []
    content = []
    
    # 将 BASE_TEXT_PROMPTS 添加到 message 的最前面
    base_prompt = BASE_TEXT_PROMPTS.format(
        text=text,
        compound=compound
    )
    content.append({"type": "text", "text": base_prompt + "\n"})
    
    # 添加 text 和 compound
    if text is not None:
        text_content_dic = {"type": "text", "text": f"Text: {text}\n"}
        content.append(text_content_dic)
    if compound is not None:
        compound_content_dic = {"type": "text", "text": f"Compound: {compound}\n"}
        content.append(compound_content_dic)
    
    # 添加图片（Base64 编码）
    if images is not None:
        for i, image_path in enumerate(images, start=1):
            image_base64 = image_to_base64(image_path)
            pic_content_dic = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            content.append(pic_content_dic)
    
    messages.append({"role": "user", "content": content})
    return messages

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(file_path, data):
    """保存 JSON 文件"""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def prepare_dataset(data):
    ds = []
    for d in data:
        # 提取图片路径并添加前缀
        image_paths = [
            f"/root/autodl-tmp/photo/{d.get('image1', '')}",
            f"/root/autodl-tmp/photo/{d.get('image2', '')}",
            f"/root/autodl-tmp/photo/{d.get('image3', '')}",
            f"/root/autodl-tmp/photo/{d.get('image4', '')}",
            f"/root/autodl-tmp/photo/{d.get('image5', '')}"
        ]

        # 提取其他字段
        text = d.get("text", "")
        compound = d.get("compound", "")

        # 构造新的数据项格式
        new_item = {
            'text': text,
            'compound': compound,
            'image1': image_paths[0],
            'image2': image_paths[1],
            'image3': image_paths[2],
            'image4': image_paths[3],
            'image5': image_paths[4]
        }
        ds.append(new_item)
    
    return ds

def main(ds_path, save_path, model_name, seed, temperature, top_p):
    # 加载数据
    lst = load_json(ds_path)
    ds = prepare_dataset(lst)
    
    res = []
    for d in tqdm(ds, desc="Processing items"):
        print(f"Processing item: {d['compound']}")
        
        # 格式化文本提示，包含短语和定义
        messages = get_message(
            images=[d.get(f"image{i+1}") for i in range(5)],  # 提取 image1 到 image5
            text=d['text'],
            compound=d['compound'],
        )
        
        try:
            # 调用 OpenAI API 进行推理
            response = client.chat.completions.create(
                model=model_name,  # 使用传入的模型名称
                messages=messages,
                stream=False,
                seed=seed,  # 设置 seed
                temperature=temperature,  # 设置 temperature
                top_p=top_p  # 设置 top_p
            )
            
            # 获取生成的文本
            output_text = response.choices[0].message.content
            
            # 将生成的文本添加到结果中
            d['order'] = output_text
            res.append(d)
        except Exception as e:
            if "data_inspection_failed" in str(e):
                print(f"Skipping item due to sensitive content: {d['compound']}")
                d['order'] = "Sensitive content detected"
            else:
                print(f"Error processing item {d['compound']}: {e}")
                d['order'] = str(e)
            res.append(d)

    # 保存结果
    save_json(save_path, res)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ds_path", required=True, help="Path to the input dataset JSON file")
    parser.add_argument("--save_path", required=True, help="Path to save the processed JSON file")
    parser.add_argument("--model_name", required=True, help="Model name for OpenAI API")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1, help="Top-p sampling")
    args = parser.parse_args()
    main(args.ds_path, args.save_path, args.model_name, args.seed, args.temperature, args.top_p)
