import pandas as pd

# 读取 JSONL 文件，注意每行是一个独立的 JSON 对象
df = pd.read_json('/data1/bowei/if-RL/fork/if-verl/data/val_set.jsonl', orient='records', lines=True)

# 将 DataFrame 转换为 Parquet 文件
df.to_parquet('/data1/bowei/if-RL/fork/if-verl/data/val_set.parquet', engine='pyarrow')  # 或使用 'fastparquet' 作为引擎
