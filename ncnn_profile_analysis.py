import re
from collections import defaultdict

# 读取 profile.txt 文件
with open("/home/finnox/a133_proflie_thread2.txt", "r", encoding="utf-8") as f:
    log_text = f.read()

# 正则匹配：OpType, LayerName, Time(ms)
pattern = re.compile(r'(\S+)\s+(\S+)\s+([\d.]+)ms')

op_time = defaultdict(float)
layer_time = {}

for op, layer, t in pattern.findall(log_text):
    t = float(t)
    op_time[op] += t
    layer_time[layer] = t

# 排序输出：按耗时降序
print("=== 各操作符总耗时 ===")
for op, t in sorted(op_time.items(), key=lambda x: x[1], reverse=True):
    print(f"{op:<15} {t:.2f} ms")

print("\n=== 各层耗时 ===")
for layer, t in sorted(layer_time.items(), key=lambda x: x[1], reverse=True):
    print(f"{layer:<20} {t:.2f} ms")

