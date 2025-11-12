# # # #
# import pandas as pd
# import json
#
#
# def process_kg_file(input_file, output_file, x_type_filter, y_type_filter, filter_on):
#     """
#     处理 kg.csv 文件，筛选符合条件的三元组，并按照头实体进行分组，确保所有键和值均为字符串格式。
#
#     :param input_file: 输入 CSV 文件路径
#     :param output_file: 输出 JSON 文件路径
#     :param x_type_filter: 头实体类型过滤条件
#     :param y_type_filter: 尾实体类型过滤条件
#     :param filter_on: 按哪个索引进行筛选 ('x_index' 或 'y_index')
#     """
#     # 读取 kg.csv 文件
#     df = pd.read_csv(input_file)
#     df = df[df["display_relation"] != "off-label use"]
#
#     # 筛选符合条件的数据
#     filtered_df = df[(df["x_type"] == x_type_filter) & (df["y_type"] == y_type_filter)]
#
#     # 统计 filter_on 指定的索引出现的次数
#     index_counts = filtered_df[filter_on].value_counts()
#
#     # 筛选出出现次数在 5-20 之间的索引
#     valid_index = index_counts[(index_counts >= 45) & (index_counts <= 50)].index
#
#     # 只保留符合条件的索引
#     final_df = filtered_df[filtered_df[filter_on].isin(valid_index)]
#
#     # 组织数据，按照头实体分组（转换为字符串）
#     entity_dict = {}
#     for row in final_df.itertuples(index=False):
#         x_index = str(row.x_index)  # 转换为字符串
#         display_relation = str(row.display_relation)
#         y_index = str(row.y_index)
#
#         if x_index not in entity_dict:
#             entity_dict[x_index] = []
#
#         # 确保三元组的每个部分都是字符串
#         entity_dict[x_index].append([x_index, display_relation, y_index])
#
#     # 统计包含多个三元组的头实体
#     multiple_triples = {key: val for key, val in entity_dict.items() if len(val) > 1}
#
#     # 打印包含多个三元组的头实体
#     print(f"【{output_file}】中包含多个三元组的头实体:")
#     for entity, triples in multiple_triples.items():
#         print(f"  {entity}: {len(triples)} 个三元组")
#
#     # 保存为 JSON 文件（确保所有部分都为字符串）
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(entity_dict, f, ensure_ascii=False, indent=4)
#
#     print(f"✅ 已成功保存 {len(entity_dict)} 个头实体的数据到 {output_file}\n")
# # # # # # #
# # # # # # # #
# # # # # # # # ✅ 处理 test_task.json (疾病 -> 药物)
# process_kg_file(
#     "./primkg/kg.csv",
#     "Y:/primkg-assistant(disease)/test_tasks4.json",
#     x_type_filter="disease",
#     y_type_filter="drug",
#     filter_on="x_index"
# )
#
# # ✅ 处理 test_task_inv.json (药物 -> 疾病)
# process_kg_file(
#     "./primkg/kg.csv",
#     "Y:/primkg-assistant(disease)/test_tasks_inv4.json",
#     x_type_filter="drug",
#     y_type_filter="disease",
#     filter_on="y_index"
# )
# ---------------------合并这两个文件---------

# import json
#
# with open("./primkg-assistant(disease)/test_tasks.json", "r") as f1, open("./primkg-assistant(disease)/test_tasks_inv.json", "r") as f2:
#     tasks1 = json.load(f1)
#     tasks2 = json.load(f2)
#
# # 合并两个 dict（默认 key 是疾病 id，值是三元组列表）
# merged_tasks = {}
#
# for k, v in tasks1.items():
#     merged_tasks[k] = v
#
# for k, v in tasks2.items():
#     if k in merged_tasks:
#         merged_tasks[k].extend(v)
#     else:
#         merged_tasks[k] = v
#
# # 保存合并后的文件
# with open("test_tasks_merged.json", "w", encoding="utf-8") as f:
#     json.dump(merged_tasks, f, ensure_ascii=False, indent=4)
#
# print(f"✅ 合并完成，共包含 {len(merged_tasks)} 个疾病任务")

# -------------------------------------------traing_tasks.py-包括了所有类型的train_tasks.json文件划分
# import pandas as pd
# import json
#
# # **定义文件路径**
# kg_file = "./primkg/kg.csv"
# test_task_file = "Y:/primkg-assistant(disease)/test_tasks4.json"
# test_task_inv_file = "Y:/primkg-assistant(disease)/test_tasks_inv4.json"
# train_task_file = "Y:/primkg-assistant(disease)/train_tasks4.json"
#
# # **读取 kg.csv 数据**
# print("📌 开始读取 kg.csv 文件...")
# kg_df = pd.read_csv(kg_file)
# kg_triples = set()
#
# for _, row in kg_df.iterrows():
#     x_index = str(row["x_index"])  # 转换为字符串
#     relation = str(row["display_relation"])
#     y_index = str(row["y_index"])
#     kg_triples.add((x_index, relation, y_index))
#
# print(f"✅ 从 kg.csv 提取三元组 {len(kg_triples)} 个")
#
# # **读取 test_tasks.json 文件**
# test_triples = set()
# test_heads = set()
#
# print("📌 开始读取 test_tasks.json 文件...")
# try:
#     with open(test_task_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         for head, triples in data.items():
#             test_heads.add(str(head))  # 记录 test_tasks.json 中的头实体
#             for triple in triples:
#                 test_triples.add(tuple(map(str, triple)))  # 转换为字符串形式的三元组
# except FileNotFoundError:
#     print(f"⚠️ 文件 {test_task_file} 未找到，跳过...")
# except json.JSONDecodeError:
#     print(f"❌ 解析 {test_task_file} 失败，跳过...")
#
# print(f"✅ 读取完成，测试集三元组 {len(test_triples)} 个，涉及头实体 {len(test_heads)} 个")
#
# # **读取 test_tasks_inv.json 文件**
# print("📌 开始读取 test_tasks_inv.json 文件...")
# try:
#     with open(test_task_inv_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         for head, triples in data.items():
#             for triple in triples:
#                 test_triples.add(tuple(map(str, triple)))  # 转换为字符串形式的三元组
# except FileNotFoundError:
#     print(f"⚠️ 文件 {test_task_inv_file} 未找到，跳过...")
# except json.JSONDecodeError:
#     print(f"❌ 解析 {test_task_inv_file} 失败，跳过...")
#
# print(f"✅ 读取完成，测试集 (含 inv) 三元组总数 {len(test_triples)} 个")
#
# # **删除在 test_task.json 和 test_tasks_inv.json 中存在的三元组**
# filtered_triples = [t for t in kg_triples if t not in test_triples]
#
# # **确保 train_tasks.json 中的头实体和尾实体不属于 test_tasks.json 文件中的头实体**
# filtered_triples = [
#     t for t in filtered_triples
#     if t[0] not in test_heads and t[2] not in test_heads
# ]
#
# print(f"✅ 过滤后剩余三元组 {len(filtered_triples)} 个")
#
# # **构建字典格式**
# train_tasks = {}
# for e1, rel, e2 in filtered_triples:
#     train_tasks.setdefault(e1, []).append([e1, rel, e2])
#
# # **写入 train_tasks.json**
# print(f"📌 开始写入 {train_task_file} ...")
# with open(train_task_file, 'w', encoding='utf-8') as f:
#     json.dump(train_tasks, f, indent=2, ensure_ascii=False)
#
# print(f"🎯 {train_task_file} 生成完毕，共 {len(filtered_triples)} 个三元组，头实体 {len(train_tasks)} 个！")




# ------------------------------- traing_tasks.py-包括了只包括了药物和疾病类型的train_tasks.json文件划分
# import pandas as pd
# import json
#
# # === 文件路径定义 ===
# kg_file = "./primkg/kg.csv"
# test_task_file = "./primkg-assistant(disease)/test_tasks1.json"
# test_task_inv_file = "./primkg-assistant(disease)/test_tasks_inv1.json"
# train_task_file = "./primkg-assistant(disease)/train_tasks1.json"
# train_task_w_file = "./primkg-assistant(disease)/train_tasks_w.json"
#
# # === 读取 kg.csv 数据（保留实体类型）===
# print("📌 开始读取 kg.csv 文件...")
# kg_df = pd.read_csv(kg_file)
#
# # 将三元组转换为 (x_index, display_relation, y_index) 并保留类型信息
# kg_triples = set()
# kg_types = {}
#
# for _, row in kg_df.iterrows():
#     x = str(row["x_index"])
#     rel = str(row["display_relation"])
#     y = str(row["y_index"])
#     x_type = str(row["x_type"])
#     y_type = str(row["y_type"])
#
#     kg_triples.add((x, rel, y))
#     kg_types[(x, rel, y)] = (x_type, y_type)
#
# print(f"✅ 从 kg.csv 提取三元组 {len(kg_triples)} 个")
#
# # === 读取 test_tasks.json 文件中的三元组和头实体 ===
# test_triples = set()
# test_heads = set()
#
# print("📌 开始读取 test_tasks.json 文件...")
# try:
#     with open(test_task_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         for head, triples in data.items():
#             test_heads.add(str(head))
#             for triple in triples:
#                 test_triples.add(tuple(map(str, triple)))
# except Exception as e:
#     print(f"⚠️ 读取 {test_task_file} 失败：{e}")
#
# # === 读取 test_tasks_inv.json 文件 ===
# print("📌 开始读取 test_tasks_inv.json 文件...")
# try:
#     with open(test_task_inv_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         for head, triples in data.items():
#             for triple in triples:
#                 test_triples.add(tuple(map(str, triple)))
# except Exception as e:
#     print(f"⚠️ 读取 {test_task_inv_file} 失败：{e}")
#
# print(f"✅ 测试集中总共有 {len(test_triples)} 个三元组，涉及头实体 {len(test_heads)} 个")
#
# # === 删除在 test_task.json 中已存在的三元组 ===
# filtered_triples = [t for t in kg_triples if t not in test_triples]
#
# # === 删除头实体或尾实体在测试集头实体中的三元组 ===
# filtered_triples = [t for t in filtered_triples if t[0] not in test_heads and t[2] not in test_heads]
#
# print(f"✅ 过滤后剩余三元组 {len(filtered_triples)} 个")
#
# # === 根据类型划分：疾病/药物相关的放入 train_tasks，其余放入 train_tasks_w ===
# train_tasks = {}
# train_tasks_w = {}
#
# for triple in filtered_triples:
#     x, rel, y = triple
#     x_type, y_type = kg_types[triple]
#
#     is_related = (x_type in {"disease", "drug"} or y_type in {"disease", "drug"})
#
#     if is_related:
#         train_tasks.setdefault(x, []).append([x, rel, y])
#     else:
#         train_tasks_w.setdefault(x, []).append([x, rel, y])
#
# # === 写入 JSON 文件 ===
# with open(train_task_file, 'w', encoding='utf-8') as f:
#     json.dump(train_tasks, f, indent=2, ensure_ascii=False)
# print(f"✅ {train_task_file} 写入完成，头实体数：{len(train_tasks)}")
#
# with open(train_task_w_file, 'w', encoding='utf-8') as f:
#     json.dump(train_tasks_w, f, indent=2, ensure_ascii=False)
# print(f"✅ {train_task_w_file} 写入完成，头实体数：{len(train_tasks_w)}")
#
# print(
#     f"🎯 总共划分三元组：相关（{sum(len(v) for v in train_tasks.values())}），无关（{sum(len(v) for v in train_tasks_w.values())}）")



# # #
# import json
# import random
#
# # 定义输入和输出文件路径
# input_file = "Y:/primkg-assistant(disease)/test_tasks4.json"
# test_output_file = "Y:/primkg-assistant(disease)/test_tasks4.json"
# dev_output_file = "Y:/primkg-assistant(disease)/dev_tasks4.json"
#
# # 读取 test_tasks.json 文件
# print("📌 开始读取 test_tasks.json...")
# with open(input_file, 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# # 将头实体打乱，确保随机性
# all_heads = list(data.keys())
# random.shuffle(all_heads)
#
# # 计算划分比例（6:4）
# test_size = int(len(all_heads) * 0.7)
# test_heads = all_heads[:test_size]  # 选取前 60% 的头实体
# dev_heads = all_heads[test_size:]   # 剩余 40% 的头实体
#
# # **构建新数据集**
# test_data = {head: data[head] for head in test_heads}
# dev_data = {head: data[head] for head in dev_heads}
#
# # **保存 test_task.json 文件**
# print(f"✅ 保存 {test_output_file}，共包含 {len(test_data)} 个头实体")
# with open(test_output_file, 'w', encoding='utf-8') as f:
#     json.dump(test_data, f, indent=2, ensure_ascii=False)
#
# # **保存 dev_tasks.json 文件**
# print(f"✅ 保存 {dev_output_file}，共包含 {len(dev_data)} 个头实体")
# with open(dev_output_file, 'w', encoding='utf-8') as f:
#     json.dump(dev_data, f, indent=2, ensure_ascii=False)
#
# print("🎯 数据划分完成！")

import json

# def load_json(file):
#     """加载 JSON 文件"""
#     try:
#         with open(file, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except FileNotFoundError:
#         print(f"文件 {file} 未找到，跳过加载。")
#         return {}
#
# def save_json(data, file):
#     """保存数据到 JSON 文件"""
#     with open(file, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)
#
# def generate_e1rel_e2(files, output_file):
#     """生成 e1rel_e2.json 文件"""
#     e1rel_e2 = {}
#
#     for file in files:
#         data = load_json(file)
#
#         for e1, triples in data.items():
#             for e1, rel, e2 in triples:
#                 key = f"{e1}{rel}"  # 组合键 e1 + rel
#                 if key not in e1rel_e2:
#                     e1rel_e2[key] = set()  # 使用集合去重
#                 e1rel_e2[key].add(e2)
#
#     # 转换集合为列表，确保 JSON 可序列化
#     e1rel_e2 = {key: list(values) for key, values in e1rel_e2.items()}
#
#     # 保存为 JSON
#     save_json(e1rel_e2, output_file)
#     print(f"文件 {output_file} 已生成，共 {len(e1rel_e2)} 个键。")
#
# # 需要处理的 JSON 文件
# # input_files = ["Y:/primkg-assistant(disease)/test_tasks4.json", "Y:/primkg-assistant(disease)/train_tasks4.json", "Y:/primkg-assistant(disease)/dev_tasks4.json"]
# input_files = ['Y:/primkg-assistant(disease)/train_tasks-rare-pro.json','Y:/primkg-assistant(disease)/alcohol withdrawal delirium_indication_test_tasks.json','Y:/primkg-assistant(disease)/delirium_contraindication_test_tasks.json','Y:/primkg-assistant(disease)/alcohol withdrawal delirium_contraindication_test_tasks.json']
# output_file = "Y:/primkg-assistant(disease)/e1rel_e2-rare-pro.json"
# #
# generate_e1rel_e2(input_files, output_file)

# import json
# import json
# from collections import defaultdict
#
# def generate_combined_train_tasks(current_test_file, other_test_files, rare_train_file, output_file):
#     # 加载 rare train tasks
#     with open(rare_train_file, 'r', encoding='utf-8') as f:
#         rare_tasks = json.load(f)
#
#     # 加载当前 test 的头实体列表
#     with open(current_test_file, 'r', encoding='utf-8') as f:
#         current_test_heads = set(json.load(f).keys())
#
#     # 加载其他 test_tasks 的内容
#     merged_test_tasks = defaultdict(list)
#     for test_file in other_test_files:
#         with open(test_file, 'r', encoding='utf-8') as f:
#             task = json.load(f)
#             for head, triples in task.items():
#                 merged_test_tasks[head].extend(triples)
#
#     # 合并 rare_tasks（排除当前 test 的 head）
#     combined_tasks = defaultdict(list)
#     for head, triples in rare_tasks.items():
#         if head not in current_test_heads:
#             combined_tasks[head].extend(triples)
#
#     # 合并其他 test_tasks 内容
#     for head, triples in merged_test_tasks.items():
#         combined_tasks[head].extend(triples)
#
#     # 写入结果
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(combined_tasks, f, indent=2)
#
#     print(f"✔ Saved: {output_file}")
#
#
# # ============ 执行生成三个文件 ============
#
# rare_file = "Y:/primkg-assistant(disease)/train_tasks-rare-pro.json"
# t1 = "Y:/primkg-assistant(disease)/alcohol withdrawal delirium_indication_test_tasks.json"
# t2 = "Y:/primkg-assistant(disease)/alcohol withdrawal delirium_contraindication_test_tasks.json"
# t3 = "Y:/primkg-assistant(disease)/delirium_contraindication_test_tasks.json"
#
# generate_combined_train_tasks(
#     current_test_file=t1,
#     other_test_files=[t2, t3],
#     rare_train_file=rare_file,
#     output_file="Y:/primkg-assistant(disease)/train_tasks-alcohol withdrawal delirium_indication.json"
# )
#
# generate_combined_train_tasks(
#     current_test_file=t2,
#     other_test_files=[t1, t3],
#     rare_train_file=rare_file,
#     output_file="Y:/primkg-assistant(disease)/train_tasks-alcohol withdrawal delirium_contraindication.json"
# )
#
# generate_combined_train_tasks(
#     current_test_file=t3,
#     other_test_files=[t1, t2],
#     rare_train_file=rare_file,
#     output_file="Y:/primkg-assistant(disease)/train_tasks-delirium_contraindication.json"
# )


# import json
# import random
#
# # 定义输入文件路径
# input_files = [
#     "./primkg-assistant(disease)/test_tasks.json",
#     # "./primkg-assistant(disease)/test_tasks_inv.json",
#     "./primkg-assistant(disease)/train_tasks.json",
#     "./primkg-assistant(disease)/dev_tasks.json"
# ]
# output_file = './primkg-assistant(disease)/e1candidates.json'
#
# # 构建头实体到尾实体的映射
# entity_to_tails = {}
# all_entities = set()
#
# # 读取并构建实体映射
# print("开始读取 JSON 文件...")
# for input_file in input_files:
#     try:
#         with open(input_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             for head_entity, triples in data.items():
#                 head_entity = str(head_entity)  # 确保是字符串
#                 entity_to_tails.setdefault(head_entity, set())
#
#                 for _, _, tail in triples:
#                     tail = str(tail)
#                     entity_to_tails[head_entity].add(tail)
#                     all_entities.add(tail)
#
#                 all_entities.add(head_entity)
#     except json.JSONDecodeError as e:
#         print(f"解析 {input_file} 时出错: {e}")
#
# print("JSON 文件读取完成，总实体数:", len(all_entities))
#
# # 转换为列表以便随机采样
# all_entities = list(all_entities)
#
# # 逐步扩充每个头实体的候选尾实体集合
# print("开始扩充候选尾实体...")
# processed = 0
# total = len(entity_to_tails)
#
# final_candidates = {}
#
# for head_entity, tails in entity_to_tails.items():
#     processed += 1
#     if processed % 1000 == 0:
#         print(f"已处理 {processed}/{total} 个头实体...")
#
#     # **1. 选择 5 个正样本尾实体**
#     # positive_tails = list(tails)
#     # num_positive = min(5, len(positive_tails))  # 如果正样本不足 5 个，则全部保留
#     # selected_positives = random.sample(positive_tails, num_positive)
#
#     # **2. 选择 15 个负样本尾实体**
#     available_tails = [e for e in all_entities if e not in tails]
#     num_negative = min(20, len(available_tails))  # 确保不会超出可选范围
#     selected_negatives = random.sample(available_tails, num_negative)
#
#     # **3. 组合正负样本，确保总数为 20**
#     final_candidates[head_entity] = selected_negatives
#
# print("候选尾实体扩充完成！")
#
# # 写入文件
# print(f"开始写入 {output_file} ...")
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(final_candidates, f, indent=2, ensure_ascii=False)
#
# print(f"{output_file} 生成完毕！")

# import json
#
# # 定义输入文件
# input_files = [
#     "./primkg-assistant(disease)/test_tasks.json",
#     # "./primkg-assistant(disease)/test_tasks_inv.json",
#     "./primkg-assistant(disease)/train_tasks.json",
#     "./primkg-assistant(disease)/dev_tasks.json"
# ]
#
# # 存储实体的集合（头实体和尾实体统一去重）
# entities = set()
#
# # 遍历所有输入文件，提取所有唯一的实体
# print("📌 开始遍历数据文件...")
# for file in input_files:
#     try:
#         with open(file, "r", encoding="utf-8") as f:
#             data = json.load(f)
#             for head, triples in data.items():
#                 entities.add(str(head))  # 头实体加入集合
#                 for triple in triples:
#                     _, _, tail = map(str, triple)  # 确保转换为字符串
#                     entities.add(tail)  # 尾实体加入集合
#     except FileNotFoundError:
#         print(f"⚠️ 文件 {file} 未找到，跳过...")
#     except json.JSONDecodeError:
#         print(f"❌ 解析 {file} 失败，跳过...")
#
# print(f"✅ 实体总数: {len(entities)}")
#
# # **编号**
# entity_map = {entity: idx for idx, entity in enumerate(sorted(entities))}
#
# # **保存为 JSON 文件**
# output_file = "./primkg-assistant(disease)/ent2ids.json"
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(entity_map, f, indent=2, ensure_ascii=False)
#
# print(f"🎯 文件 {output_file} 生成完毕！")




# import json
# from collections import defaultdict
#
# # 定义输入和输出文件路径
# test_file = "./primkg-assistant(disease)/test_tasks.json"
# train_file = "./primkg-assistant(disease)/train_tasks.json"
# output_file = "./primkg-assistant(disease)/matching_triples.json"
#
# # 读取 test_task.json 文件
# print("📌 正在读取 test_task.json...")
# with open(test_file, 'r', encoding='utf-8') as f:
#     test_data = json.load(f)
#
# # 读取 train_tasks.json 文件
# print("📌 正在读取 train_tasks.json...")
# with open(train_file, 'r', encoding='utf-8') as f:
#     train_data = json.load(f)
#
# # ✅ 构建快速查询集合
# train_heads = set(train_data.keys())  # 将 train 的头实体转换为集合（快速匹配）
# train_tails = set()
# train_triples_map = defaultdict(list)
#
# # 遍历 train_tasks.json，建立映射（加快查找速度）
# for head, triples in train_data.items():
#     for triple in triples:
#         e1, rel, e2 = map(str, triple)
#         train_tails.add(e2)  # 建立尾实体集合
#         train_triples_map[e2].append([e1, rel, e2])  # 按照尾实体建立索引
#         train_triples_map[e1].append([e1, rel, e2])  # 头实体也建立索引（便于查找）
#
# print(f"✅ 训练集头实体数量: {len(train_heads)}")
# print(f"✅ 训练集尾实体数量: {len(train_tails)}")
#
# # ✅ 匹配 test_task.json 文件的尾实体
# result = defaultdict(list)
#
# for head, triples in test_data.items():
#     for triple in triples:
#         _, _, test_tail = map(str, triple)
#         if test_tail in train_heads or test_tail in train_tails:
#             # 如果 test_tail 匹配到 train_tasks.json 的头实体或尾实体，保存匹配三元组
#             if test_tail in train_triples_map:
#                 result[test_tail].extend(train_triples_map[test_tail])
#
# # ✅ 去重（防止同一三元组多次匹配）
# for key in result:
#     result[key] = [list(x) for x in set(tuple(x) for x in result[key])]
#
# # ✅ 保存匹配结果
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(result, f, indent=2, ensure_ascii=False)
#
# print(f"🎯 匹配结果已保存到 {output_file}，共匹配到 {len(result)} 个尾实体！")



# 检查是否有头实体在训练集中或尾实体在训练集中
# import json
#
# # 定义文件路径
# test_file = "./primkg-assistant(disease)/test_tasks.json"
# train_file = "./primkg-assistant(disease)/train_tasks.json"
#
# # 读取 test_task.json 文件
# print("📌 正在读取 test_task.json...")
# with open(test_file, 'r', encoding='utf-8') as f:
#     test_data = json.load(f)
#
# # 读取 train_tasks.json 文件
# print("📌 正在读取 train_tasks.json...")
# with open(train_file, 'r', encoding='utf-8') as f:
#     train_data = json.load(f)
#
# # ✅ 构建快速查询集合
# train_heads = set(train_data.keys())  # 训练集的头实体集合
# train_tails = set()
#
# # 遍历 train_tasks.json，建立尾实体集合
# for triples in train_data.values():
#     for triple in triples:
#         _, _, tail = map(str, triple)
#         train_tails.add(tail)  # 建立尾实体集合
#
# print(f"✅ 训练集头实体数量: {len(train_heads)}")
# print(f"✅ 训练集尾实体数量: {len(train_tails)}")
#
# # ✅ 匹配 test_task.json 中的头实体
# matched_heads = {}
#
# for head in test_data.keys():
#     if head in train_heads:
#         matched_heads[head] = "在训练集头实体中"
#     elif head in train_tails:
#         matched_heads[head] = "在训练集尾实体中"
#
# # ✅ 打印匹配结果
# if matched_heads:
#     print("\n🎯 匹配到的头实体如下：")
#     for head, position in matched_heads.items():
#         print(f"头实体: {head} ➔ {position}")
# else:
#     print("\n🚫 未找到匹配的头实体")
#
# print(f"\n✅ 匹配到 {len(matched_heads)} 个头实体！")

import json

# def count_tasks(file_path):
#     with open(file_path, 'r') as f:
#         tasks = json.load(f)
#     num_heads = len(tasks.keys())
#     num_triples = sum(len(triples) for triples in tasks.values())
#     return num_heads, num_triples
#
# train_heads, train_triples = count_tasks('./primkg-assistant(disease)/train_tasks.json')
# dev_heads, dev_triples = count_tasks('./primkg-assistant(disease)/dev_tasks.json')
# # 如果有 test_tasks.json 文件，也可以统计：
# test_heads, test_triples = count_tasks('./primkg-assistant(disease)/test_tasks.json')
#
# print("train_tasks.json: {} head entities, {} triples".format(train_heads, train_triples))
# print("dev_tasks.json: {} head entities, {} triples".format(dev_heads, dev_triples))
# print("test_tasks.json: {} head entities, {} triples".format(test_heads, test_triples))
#


# import json
#
# def count_tasks_with_relations_and_tails(file_path):
#     with open(file_path, 'r') as f:
#         tasks = json.load(f)
#
#     num_heads = len(tasks)
#     num_triples = 0
#     indication_count = 0
#     contraindication_count = 0
#     tail_entities = set()
#
#     for triples in tasks.values():
#         num_triples += len(triples)
#         for triple in triples:
#             rel = triple[1]
#             tail = triple[2]
#             tail_entities.add(tail)
#             if rel == 'indication':
#                 indication_count += 1
#             elif rel == 'contraindication':
#                 contraindication_count += 1
#
#     num_tail_entities = len(tail_entities)
#     return num_heads, num_triples, indication_count, contraindication_count, num_tail_entities
#
# # 使用函数
# # train_heads, train_triples, train_ind, train_contra, train_tails = count_tasks_with_relations_and_tails('Y:/primkg-assistant(disease)/train_tasks2.json')
# # dev_heads, dev_triples, dev_ind, dev_contra, dev_tails = count_tasks_with_relations_and_tails('Y:/primkg-assistant(disease)/dev_tasks4.json')
# test_heads, test_triples, test_ind, test_contra, test_tails = count_tasks_with_relations_and_tails('Y:/primkg-assistant(disease)/test_tasks4.json')
# #
# # print(f"train_tasks.json: {train_heads} head entities, {train_triples} triples, {train_tails} unique tail entities")
# # print(f"  - indication: {train_ind}, contraindication: {train_contra}")
#
# # print(f"dev_tasks.json: {dev_heads} head entities, {dev_triples} triples, {dev_tails} unique tail entities")
# # print(f"  - indication: {dev_ind}, contraindication: {dev_contra}")
#
# print(f"test_tasks.json: {test_heads} head entities, {test_triples} triples, {test_tails} unique tail entities")
# print(f"  - indication: {test_ind}, contraindication: {test_contra}")


import pandas as pd
import json

def create_train_test_split(kg_file, train_output, test_output, test_heads, test_relations):
    df = pd.read_csv(kg_file)
    df = df[df["display_relation"] != "off-label use"]

    df["x_index"] = df["x_index"].astype(str)
    df["y_index"] = df["y_index"].astype(str)
    df["display_relation"] = df["display_relation"].astype(str)
    df["x_name"] = df["x_name"].astype(str)

    # ✅ 添加多重条件过滤：x_type, x_name, display_relation
    test_df = df[
        (df["x_type"] == "disease") &
        (df["x_name"].isin(test_heads)) &
        (df["display_relation"].isin(test_relations))
    ]

    train_df = df.drop(index=test_df.index)

    def build_triple_dict(df):
        triple_dict = {}
        for row in df.itertuples(index=False):
            head = row.x_index
            rel = row.display_relation
            tail = row.y_index
            triple = [head, rel, tail]
            if head not in triple_dict:
                triple_dict[head] = []
            triple_dict[head].append(triple)
        return triple_dict

    test_triples = build_triple_dict(test_df)
    train_triples = build_triple_dict(train_df)

    inverse_set = set()
    for triples in test_triples.values():
        for h, r, t in triples:
            inverse_set.add((t, r, h))

    cleaned_train_triples = {}
    for head, triples in train_triples.items():
        cleaned = [trip for trip in triples if (trip[2], trip[1], trip[0]) not in inverse_set]
        if cleaned:
            cleaned_train_triples[head] = cleaned

    with open(test_output, "w", encoding="utf-8") as f_test:
        json.dump(test_triples, f_test, ensure_ascii=False, indent=4)
    print(f"✅ 测试集保存成功，包含 {len(test_triples)} 个头实体：{test_output}")

    with open(train_output, "w", encoding="utf-8") as f_train:
        json.dump(cleaned_train_triples, f_train, ensure_ascii=False, indent=4)
    print(f"✅ 训练集保存成功，包含 {len(cleaned_train_triples)} 个头实体：{train_output}")

# ✅ 调用主函数
kg_file = "/home/ubuntu/YL/primkg-assistant(disease)/kg.csv"
train_output = "/home/ubuntu/YL/primkg-assistant(disease)/train_cystic fibrosis_tasks.json"
test_output = "/home/ubuntu/YL/primkg-assistant(disease)/test_cystic fibrosis_tasks.json"

test_heads = ["cystic fibrosis"]
test_relations = ["indication", "contraindication"]

create_train_test_split(kg_file, train_output, test_output, test_heads, test_relations)
