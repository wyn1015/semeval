import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from datasets import load_dataset
from tqdm import tqdm
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from functools import partial
from transformers import Trainer, TrainingArguments, set_seed
import argparse
import json
# 引入 SwanLab 回调函数
from swanlab.integration.huggingface import SwanLabCallback

class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x

def transform_images(examples, image_transformations):
    images = []
    for image_file in examples['image']:
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Image file not found: {image_file}")
        try:
            image = read_image(image_file, mode=ImageReadMode.RGB)
            if image is None:
                raise ValueError(f"Failed to read image: {image_file}")
            transformed_image = image_transformations(image)
            images.append(transformed_image)
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            # 用一个占位张量替代损坏的图像，以保持列表长度一致
            images.append(torch.zeros(3, 336, 336))  # 假设图像大小为 336x336
    examples["pixel_values"] = images
    return examples

# def process_line(examples, tokenizer, img_dir):
#     compound = examples['compound']
#     images = examples['image']
#     examples['image'] = [os.path.join(img_dir, img) for img in images]  # 直接从 IMAGE_DIR 读取图像路径
    
#     print("Generated image paths:")
#     for img_path in examples['image']:
#         print(img_path)
    
#     captions = examples.get('text', '')
#     explanations = examples['response']  # 假设你的数据集中有一个'response'字段
   

#     # 根据sentence_type处理text和explanation
#     for i, caption in enumerate(captions):
#         if examples['sentence_type'][i] == "idiomatic":
#             captions[i] = explanations[i]  # 拼接text和explanation
#     text_inputs = tokenizer(captions, max_length=77, padding="max_length", truncation=True)
#     examples["input_ids"] = text_inputs.input_ids
#     examples["attention_mask"] = text_inputs.attention_mask
#     return examples

import os
import re
import os
from collections import defaultdict
import os
import re
from collections import defaultdict

import os
import re
from collections import defaultdict

def process_line(examples, tokenizer, img_dir):
    compound = examples['compound']
    images = examples['image']
    examples['image'] = [os.path.join(img_dir, img) for img in images]  # 直接从 IMAGE_DIR 读取图像路径
    
    print("Generated image paths:")
    for img_path in examples['image']:
        print(img_path)
    
    captions = []
    for i, current_compound_list in enumerate(compound):
        # 确保 current_compound 是字符串（从单元素列表中提取）
        if isinstance(current_compound_list, list) and len(current_compound_list) == 1:
            current_compound = current_compound_list[0]  # 提取单元素列表中的字符串
        else:
            raise ValueError(f"Unexpected format for 'compound': {current_compound_list}")
        
        # 获取当前样本的 sentence_type
        sentence_type = examples['sentence_type'][i]
        
        # 从图片名称中提取数字（如 image1 中的 1）
        image_name = images[i]  # 获取图片名称
        match = re.search(r'image(\d+)', image_name)
        if match:
            image_number = int(match.group(1))  # 提取数字
            sentence_key = f'sentence{image_number}'  # 根据数字确定 sentencex 字段
        else:
            sentence_key = None  # 如果提取不到数字，设置为 None

        # 如果提取到数字，优先使用 sentenceX 字段
        if sentence_key and sentence_key in examples and examples[sentence_key][i]:
            captions.append(examples[sentence_key][i])
            print(f"Loaded caption from {sentence_key}: {examples[sentence_key][i]}")  # 打印加载的内容
        else:
            # 如果 sentenceX 字段不存在或为空，使用 text 字段
            if 'text' in examples and examples['text'][i]:
                captions.append(examples['text'][i])
                print(f"Loaded caption from 'text': {examples['text'][i]}")  # 打印加载的内容
            else:
                # 如果 text 字段也不存在，使用默认值
                captions.append("No caption available")
                print(f"No caption found for {sentence_key} or 'text', using default value.")  # 打印默认值提示
    
    # 对 captions 进行 tokenization
    text_inputs = tokenizer(captions, max_length=77, padding="max_length", truncation=True)
    examples["input_ids"] = text_inputs.input_ids
    examples["attention_mask"] = text_inputs.attention_mask
    return examples
# def process_line(examples, tokenizer, img_dir):
#     compound = examples['compound']
#     images = examples['image']
#     examples['image'] = [os.path.join(img_dir, img) for img in images]  # 直接从 IMAGE_DIR 读取图像路径
    
#     print("Generated image paths:")
#     for img_path in examples['image']:
#         print(img_path)
    
#     # 用于记录每个 compound 出现的次数
#     compound_count = defaultdict(int)
    
#     captions = []
#     for i, current_compound_list in enumerate(compound):
#         # 确保 current_compound 是字符串（从单元素列表中提取）
#         if isinstance(current_compound_list, list) and len(current_compound_list) == 1:
#             current_compound = current_compound_list[0]  # 提取单元素列表中的字符串
#         else:
#             raise ValueError(f"Unexpected format for 'compound': {current_compound_list}")
        
#         # 获取当前样本的 sentence_type
#         sentence_type = examples['sentence_type'][i]
        

#         # if sentence_type == "idiomatic":
#         if sentence_type in ["idiomatic", "literal"]:
#             # 根据 compound 出现的次数选择 sentenceX
#             compound_count[current_compound] += 1  # 增加 compound 的计数
#             sentence_num = (compound_count[current_compound] - 1) % 5 + 1  # 计算 sentenceX 的编号（1-5）
#             sentence_key = f'sentence{sentence_num}'  # 构造 sentenceX 字段名

#             # 优先使用 sentenceX 字段
#             if sentence_key in examples and examples[sentence_key][i]:
#                 captions.append(examples[sentence_key][i])
#                 print(f"Loaded caption from {sentence_key}: {examples[sentence_key][i]}")  # 打印加载的内容
#             else:
#                 # 如果 sentenceX 字段不存在或为空，使用 text 字段
#                 if 'text' in examples and examples['text'][i]:
#                     captions.append(examples['text'][i])
#                     print(f"Loaded caption from 'text': {examples['text'][i]}")  # 打印加载的内容
#                 else:
#                     # 如果 text 字段也不存在，使用默认值
#                     captions.append("No caption available")
#                     print(f"No caption found for {sentence_key} or 'text', using default value.")  # 打印默认值提示
#         else:
#             # 如果不是 idiomatic 类型，直接使用 text 字段
#             captions.append(examples['text'][i])
#             print(f"Loaded caption from 'text': {examples['text'][i]}")  # 打印加载的内容
    
#     # 对 captions 进行 tokenization
#     text_inputs = tokenizer(captions, max_length=77, padding="max_length", truncation=True)
#     examples["input_ids"] = text_inputs.input_ids
#     examples["attention_mask"] = text_inputs.attention_mask
#     return examples
    
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }

def main(args):
    # 从环境变量或命令行参数中读取随机种子
    seed = int(os.getenv('RANDOM_SEED', args.random_seed))

    # 固定随机种子
    set_seed(seed)  # 使用 Hugging Face 的 set_seed 函数
    print(f"Using random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Using random seed: {seed}")

    model_name_or_path = args.model_name_or_path
    img_dir = args.image_dir
    data_path = args.data_path

    # 从环境变量中读取 batch_size 和 num_epochs
    batch_size = int(os.getenv('BATCH_SIZE', args.batch_size))
    num_epochs = int(os.getenv('NUM_EPOCHS', args.num_epochs))

    # 提取 DATA_PATH 中的特定部分
    data_path_suffix = os.path.basename(data_path)
    data_path_parts = data_path_suffix.split('_')
    experiment_suffix = f"{data_path_parts[0]}-{data_path_parts[-1].split('.')[0]}"

    # 构建 experiment_name
    # experiment_name = f"336_243_nocheck_train_train_{num_epochs}_bs{batch_size}-{experiment_suffix}"
    experiment_name = f"127_sll_dev_train_train_{num_epochs}_bs{batch_size}-{experiment_suffix}"

    # 构建输出目录和日志目录
    # output_dir = os.path.join(args.output_dir, experiment_name)
    output_dir = args.output_dir
    logging_dir = os.path.join(args.output_dir, "logs")

    # 加载数据集
    ds = load_dataset("json", data_files={"train": data_path})
    print("Dataset structure:")
    print(ds["train"][0])  # 打印第一条数据的结构

    # 计算 warmup_steps
    total_steps = len(ds["train"]) // batch_size * num_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% 的总步骤数作为 warmup

    # 加载预训练的CLIP模型和处理器
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    image_transformations = Transform(
        336, processor.image_processor.image_mean, processor.image_processor.image_std
    )

    # 将图像变换函数转换为TorchScript
    image_transformations = torch.jit.script(image_transformations)

    # 定义数据预处理函数
    process_fn = partial(process_line, tokenizer=processor, img_dir=img_dir)
    ds = ds.map(process_fn, batched=True, batch_size=1)
    ds.set_transform(partial(transform_images, image_transformations=image_transformations))

    # 加载CLIP模型
    model = AutoModel.from_pretrained(model_name_or_path)
    
    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir=logging_dir,
        logging_steps=5,
        save_steps=280,
        save_total_limit=2,
        evaluation_strategy="no",
        eval_steps=500,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        fp16=True,  # 如果GPU支持混合精度训练
        remove_unused_columns=False,
        seed=seed,
    )
    set_seed(training_args.seed)
    print("=================================")
    print(seed)
    # 保存训练参数到 JSON 文件
    training_params = {
        "model_name_or_path": model_name_or_path,
        "image_dir": img_dir,
        "data_path": data_path,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "fp16": True,
        "output_dir": output_dir,
        "logging_dir": logging_dir,
        "logging_steps": 5,
        "save_steps": 280,
        "save_total_limit": 2,
        "evaluation_strategy": "no",
        "eval_steps": 500,
        "remove_unused_columns": False,
    }

    # 保存到 JSON 文件
    params_file_path = os.path.join(output_dir, "training_params.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(params_file_path, "w") as f:
        json.dump(training_params, f, indent=4)

    # 实例化 SwanLabCallback
    swanlab_callback = SwanLabCallback(project="en-CLIP-Training-pt-for", experiment_name=experiment_name)

    # 创建Trainer实例
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        data_collator=collate_fn,
        callbacks=[swanlab_callback],  # 添加 SwanLab 回调
    )

    # 开始训练
    trainer.train()

    # 保存微调后的模型和处理器
    model_save_path = os.path.join(output_dir, "model")
    processor_save_path = os.path.join(output_dir, "processor")


    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(processor_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    processor.save_pretrained(processor_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON dataset file")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or name of the pretrained CLIP model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for training results")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)
