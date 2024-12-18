# import torch
# print(torch.__version__)  # 查看 PyTorch 版本
# print(torch.cuda.is_available())  # 检查是否支持 CUDA
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))
# print(torch.__version__)
#
# import transformers
# print(transformers.__version__)

import transformers
import torch

# 切换为你下载的模型文件目录, 这里的demo是Llama-3-8B-Instruct
# 如果是其他模型，比如qwen，chatglm，请使用其对应的官方demo
model_id = r"C:\Users\PC\LLaMA-Factory\Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print('1',outputs[0]["generated_text"][len(prompt):])



