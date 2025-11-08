#!/usr/bin/env python3
"""
兼容非法 JSON 字符串的解析器
"""

import json
import re


def fix_malformed_json(json_str: str) -> str:
    """
    修复常见的 JSON 格式错误
    
    策略：逐字符追踪括号栈，发现不匹配时智能插入闭合括号
    """
    # 移除可能的注释
    json_str = re.sub(r'//.*?\n', '\n', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # 移除多余的逗号（在 } 或 ] 前面的逗号）
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    lines = json_str.split('\n')
    
    # 使用栈追踪括号，记录每个开括号的信息
    stack = []  # [(bracket_type, line_idx, indent)]
    in_string = False
    escape = False
    
    for line_idx, line in enumerate(lines):
        line_indent = len(line) - len(line.lstrip())
        
        for char_idx, char in enumerate(line):
            if escape:
                escape = False
                continue
            
            if char == '\\':
                escape = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if char == '{':
                stack.append(('{', line_idx, line_indent))
            elif char == '[':
                stack.append(('[', line_idx, line_indent))
            elif char == '}':
                # 尝试匹配栈顶的 {
                if stack and stack[-1][0] == '{':
                    stack.pop()
                # 如果栈顶不是 {，说明有问题，但先不处理
            elif char == ']':
                # 尝试匹配栈顶的 [
                # 但在匹配之前，检查栈中是否有未闭合的 {
                # 如果有，应该先闭合所有在这个 [ 之后打开的 {
                bracket_found = False
                temp_braces = []
                
                while stack:
                    top_type, top_line, top_indent = stack[-1]
                    if top_type == '[':
                        stack.pop()
                        bracket_found = True
                        break
                    elif top_type == '{':
                        # 这个 { 需要被闭合
                        temp_braces.append(stack.pop())
                    else:
                        break
                
                # 记录需要插入的 }
                if temp_braces:
                    # 在当前位置之前插入缺失的 }
                    for brace_type, brace_line, brace_indent in reversed(temp_braces):
                        # 在当前行之前插入 }
                        # 使用适当的缩进（应该与开括号的缩进一致）
                        lines[line_idx] = ' ' * brace_indent + '}' + '\n' + lines[line_idx]
    
    # 简单粗暴的方法：直接找到问题并修复
    # 重新分析一遍，这次直接修正
    result_lines = []
    stack = []
    in_string = False
    escape = False
    
    for line_idx, line in enumerate(lines):
        line_indent = len(line) - len(line.lstrip())
        line_content = line.strip()
        
        # 在添加这一行之前，检查是否需要闭合括号
        # 特别是当遇到 ] 或 } 时，检查栈的状态
        if line_content.startswith(']'):
            # 检查栈中是否有未闭合的 {（在最近的 [ 之后）
            temp_stack = stack.copy()
            unclosed_braces = []
            
            while temp_stack:
                top = temp_stack.pop()
                if top[0] == '[':
                    break
                elif top[0] == '{':
                    unclosed_braces.append(top)
            
            # 插入缺失的 }
            for brace_type, brace_line, brace_indent in unclosed_braces:
                result_lines.append(' ' * brace_indent + '}')
                # 同时从栈中移除
                if stack and stack[-1] == (brace_type, brace_line, brace_indent):
                    stack.pop()
        
        # 处理当前行的括号
        for char in line:
            if escape:
                escape = False
                continue
            
            if char == '\\':
                escape = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if char == '{':
                stack.append(('{', line_idx, line_indent))
            elif char == '[':
                stack.append(('[', line_idx, line_indent))
            elif char == '}':
                if stack and stack[-1][0] == '{':
                    stack.pop()
            elif char == ']':
                if stack and stack[-1][0] == '[':
                    stack.pop()
        
        result_lines.append(line)
    
    result = '\n'.join(result_lines)
    
    # 再次移除多余的逗号
    result = re.sub(r',(\s*[}\]])', r'\1', result)
    
    return result


def parse_malformed_json(json_str: str) -> dict:
    """
    尝试解析非法的 JSON 字符串
    """
    # 第一次尝试：直接解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # 第二次尝试：修复后解析
    try:
        fixed_json = fix_malformed_json(json_str)
        return json.loads(fixed_json)
    except json.JSONDecodeError as e:
        print(f"修复后的 JSON：\n{fixed_json}\n")
        print(f"解析错误：{e}")
        raise


def main():
    # 测试用的非法 JSON
    malformed_json = """{
  "horizon": "1m",
  "views": [
    {
      "type": "absolute",
      "asset": "SPY",
      "expected_return": 0.12,
      "confidence": 0.7
    },
    {
      "type": "absolute",
      "asset": "GLD",
      "expected_return": 0.15,
      "confidence": 0.8
    },
    {
      "type": "absolute",
      "asset": "TLT",
      "expected_return": 0.10,
      "confidence": 0.6
  ],
  "rationale": "",
  "timestamp": "2025-10-30"
}"""
    
    print("原始 JSON 字符串：")
    print(malformed_json)
    print("\n" + "="*50 + "\n")
    
    try:
        result = parse_malformed_json(malformed_json)
        print("解析成功！✓")
        print("\n解析结果：")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        print("\n" + "="*50 + "\n")
        print("验证数据：")
        print(f"Horizon: {result['horizon']}")
        print(f"Views 数量: {len(result['views'])}")
        for i, view in enumerate(result['views'], 1):
            print(f"  View {i}: {view['asset']} - {view['expected_return']*100}% (confidence: {view['confidence']})")
        print(f"Timestamp: {result['timestamp']}")
        
    except Exception as e:
        print(f"解析失败：{e}")


if __name__ == "__main__":
    main()
