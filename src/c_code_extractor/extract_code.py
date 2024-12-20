import click
import json
import os
from ast_analyze import Point
from typing import Sequence, Literal
from ast_analyze import get_func_header_from_def
from trace_func_depends import contain_leak, memory_container
from tqdm import tqdm

class CodeLocation:
    def __init__(self, file, start_point: Point, end_point: Point):
        self.file = file
        self.start_point = start_point
        self.end_point = end_point
        
class MergeItem:
    def __init__(self, kind: Literal['location', 'include', 'func', 'macro'], item: CodeLocation | str | tuple[str, str]) -> None:
        self.kind = kind
        self.item = item
        match kind:
            case 'location' | 'func':
                assert isinstance(item, CodeLocation)
            case 'include':
                assert isinstance(item, str)
            case 'macro':
                assert isinstance(item, tuple)
                assert isinstance(item[0], str)
                assert isinstance(item[1], str)
class Slice:
    def __init__(self, function_name: str, to_merge: Sequence[MergeItem]):
        self.function_name = function_name
        self.to_merge = to_merge
        
def extract_content(file: str, start_point: Point, end_point: Point) -> str:
    with open(file, 'r') as f:
        lines = [l.removesuffix('\n') if l.endswith('\n') else l for l in f.readlines()]
    if start_point[0] == end_point[0]:
        return lines[start_point[0]][start_point[1]:end_point[1]]
    else:
        first_line = lines[start_point[0]][start_point[1]:]
        middle_lines = lines[start_point[0]+1:end_point[0]]
        last_line = lines[end_point[0]][:end_point[1]]
        # The Tree-sitter C grammar has a bug. The trailing semicolon of `struct {...};` is not included in the range of the struct.
        if end_point[1] < len(lines[end_point[0]]) and lines[end_point[0]][end_point[1]] == ';':
            last_line += ';'
        return '\n'.join([first_line] + middle_lines + [last_line])

@click.command()
@click.option('--src', '-i', required=True, help='The source directory')
@click.option('--output', '-o', required=True, help='The output directory')
@click.option('function_depends_file', '--function-depends', '-f', required=True, help='The function depends file')
def main(src, output, function_depends_file):
    function_depends: list[dict] = []
    with open(function_depends_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            function_depends.append(item)

    func_name_repeat = {}
    for item in tqdm(function_depends):
        func_name = item['func']['name']
        to_merge = [
            (0, MergeItem('location', 
                          CodeLocation(os.path.join(src, item['func']['file']), Point(item['func']['start_point']), Point(item['func']['end_point']))))
        ]
        for depend in item['other_depends']:
            order = int(depend['order'])
            code_item = depend['code_item']
            start_point = Point(code_item['start_point'])
            end_point = Point(code_item['end_point'])
            to_merge.append((order, MergeItem('location', CodeLocation(code_item['file'], start_point, end_point))))
        for depend in item['include_depends']:
            order = int(depend['order'])
            code_item = depend['code_item']
            to_merge.append((order, MergeItem('include', code_item['file'])))
        for depend in item['func_depends']:
            order = int(depend['order'])
            code_item = depend['code_item']
            start_point = Point(code_item['start_point'])
            end_point = Point(code_item['end_point'])
            to_merge.append((order, MergeItem('func', CodeLocation(code_item['file'], start_point, end_point))))
        to_merge.sort(key=lambda x: x[0], reverse=True)
        for macro in item['macros']:
            to_merge.insert(0, (-1, MergeItem('macro', tuple(macro))))
        slice_ = Slice(func_name, [x[1] for x in to_merge])
        
        if func_name in func_name_repeat:
            func_name_repeat[func_name] += 1
            dir_name = f'{func_name}_{func_name_repeat[func_name]}'
        else:
            func_name_repeat[func_name] = 0
            dir_name = func_name
        os.makedirs(os.path.join(output, dir_name), exist_ok=True)
        includes = set()
        contents = []
        for item in slice_.to_merge:
            content = ''
            match item.kind:
                case 'location':
                    assert isinstance(item.item, CodeLocation)
                    content = extract_content(item.item.file, item.item.start_point, item.item.end_point)
                case 'include':
                    assert isinstance(item.item, str)
                    includes.add(f'#include "{item.item}"')
                    continue
                case 'macro':
                    assert isinstance(item.item, tuple)
                    original, replacement = item.item
                    content = f'#define {original} {replacement}'
                case 'func':
                    assert isinstance(item.item, CodeLocation)
                    with open(item.item.file, 'r') as f:
                        file_content = f.read()
                    header = contain_leak('get_func_header', file_content, item.item.start_point, item.item.end_point)
                    header = header.replace('static ', '') # A funciton cannot be both static and extern
                    content = 'extern ' + header + ';'
            contents.append(content)
        contents.insert(0, '\n'.join(includes))
        with open(os.path.join(output, dir_name, f'{func_name}.c'), 'w') as f:
            f.write('\n\n'.join(contents))

if __name__ == '__main__':
    main()
