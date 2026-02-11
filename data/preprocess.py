import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess():
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    ds = load_dataset('json', data_files='data/pretrain_hq.jsonl', split='train')
    
    all_ids = []
    for example in tqdm(ds, desc="Tokenizing"):
        # 预先拼接 BOS/EOS，并进行分词
        text = str(example['text'])
        ids = tokenizer.encode(text, add_special_tokens=False)
        # 将所有数据连成一片，实现 Packing
        all_ids.extend([tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id])
    
    # 转换为 numpy 数组并保存
<<<<<<< HEAD
    all_ids = np.array(all_ids, dtype=np.int32)
=======
    all_ids = np.array(all_ids, dtype=np.uint16)
>>>>>>> da8a0fd5f7950b005684b0a094219ef543e49ae8
    all_ids.tofile('data/pretrain_data.bin')
    print(f"预处理完成，共有 {len(all_ids)} tokens")

if __name__ == "__main__":
    preprocess()