import re
from modules import instructions_registry

import inspect

def call_build_description(obj, args):
    """
    1. 获取 obj.build_description 方法的参数名
    2. 从 args 里挑出匹配的参数
    3. 调用 obj.build_description 并传入筛选后的参数
    """
    # 获取 `build_description` 方法的参数信息
    method_signature = inspect.signature(obj.build_description)

    # 获取该方法真正需要的参数名
    valid_params = set(method_signature.parameters.keys())

    # 只保留 args 中的匹配参数
    filtered_args = {k: v for k, v in args.items() if k in valid_params}

    # 调用 `build_description` 方法
    return obj.build_description(**filtered_args)


def instruction_compute_score(answer, item):
    

    answer = answer.split('</think>\n\n')[-1]

    # First extract content between <answer> tags
    # answer_pattern = r'<answer>(.*?)</answer>'
    # answer_match = re.search(answer_pattern, answer, re.DOTALL)
    # if not answer_match:
    #     return 0
    
    ids_to_check = item['instruction_id_list']
    args_to_check = item['kwargs']
    resp_to_check = answer
    
    # print('=====test=====')
    # print(resp_to_check)
    # print('=====test=====')

    is_following_list = []
    for ids, arg in zip(ids_to_check, args_to_check):

        instruction_cls = instructions_registry.INSTRUCTION_DICT[ids]
        instruction = instruction_cls(ids)
        call_build_description(instruction, arg)
        
        if resp_to_check.strip() and instruction.check_following(resp_to_check):
            is_following_list.append(True)
        else:
            is_following_list.append(False)
            
    # Normalize the score to be between 0 and 1
    score = is_following_list.count(True) / len(is_following_list)
    return score

def instruction_val_compute_score(answer, item):
    
    answer = answer.split('</think>\n\n')[-1]
    # First extract content between <answer> tags
    # answer_pattern = r'<answer>(.*?)</answer>'
    # answer_match = re.search(answer_pattern, answer, re.DOTALL)
    # if not answer_match:
    #     # print("not answer_match")
    #     return 0,0,len(item),0
    
    ids_to_check = item['instruction_id_list']
    args_to_check = item['kwargs']
    # resp_to_check = answer_match.group(1)
    resp_to_check = answer
    # print('=====test=====')
    # print(resp_to_check)
    # print('=====test=====')

    is_following_list = []
    for ids, arg in zip(ids_to_check, args_to_check):

        instruction_cls = instructions_registry.INSTRUCTION_DICT[ids]
        instruction = instruction_cls(ids)
        call_build_description(instruction, arg)
        
        if resp_to_check.strip() and instruction.check_following(resp_to_check):
            is_following_list.append(True)
        else:
            is_following_list.append(False)
            
    # Normalize the score to be between 0 and 1
    score = is_following_list.count(True) / len(is_following_list)

    all_right=0
    if score == 1:
        all_right=1

    return all_right,is_following_list.count(True),len(is_following_list),score
