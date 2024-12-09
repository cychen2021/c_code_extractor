import ast_analyze as aa
from ast_analyze import NoASTError, Point
from concurrent.futures import ProcessPoolExecutor
from defs import PROJECT_ROOT
from lsp_server import uri_to_path
from code_item import CodeItem
from lsp_client import ClangD, Edit
import random
from typing import Sequence, Literal, Any
import os.path

memory_container = ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1)

def random_permutation[T](items: list[T]) -> list[T]:
    return random.sample(items, len(items))

def line_column_to_offset(line: int, column: int, line_sizes: Sequence[int]) -> int:
    return sum(line_sizes[:line]) + column

def offset_to_line_column(offset: int, line_sizes: Sequence[int]) -> tuple[int, int]:
    line = 0
    while line < len(line_sizes) and offset >= line_sizes[line]:
        offset -= line_sizes[line]
        line += 1
    return line, offset

def cancel_macro(content: str, start_point: Point, end_point: Point) -> str:
    return contain_leak('cancel_macro', content, start_point, end_point)

def contain_leak(function_name: str, content_or_file: str, start_point: Point, end_point: Point) -> Any:
    match function_name:
        case 'get_code_item':
            f = memory_container.submit(aa.get_code_item, content_or_file, start_point, end_point)
        case 'cancel_macro':
            f = memory_container.submit(aa.cancel_macro, content_or_file, start_point, end_point)
        case _:
            f = memory_container.submit(aa.leaking_wrapper, function_name, content_or_file, start_point, end_point)
    return f.result()

def apply_edits(content: str, edits: Sequence[Edit], points: Sequence[Point]) -> tuple[str, Sequence[Point]]:
    if not edits:
        return content, points
    
    line_sizes = [len(l) + 1 for l in content.split('\n')]
    point_offsets = [line_column_to_offset(p[0], p[1], line_sizes) for p in points]
    point_offsets.sort()
    
    all_edits = [(
        edit, 
        line_column_to_offset(*edit.start_point, line_sizes),          
        line_column_to_offset(*edit.end_point, line_sizes)
    ) for edit in edits]
    all_edits.sort(key=lambda edit: edit[1])

    for i in range(1, len(all_edits)):
        assert all_edits[i-1][2] <= all_edits[i][1]
    
    mapping = []
    change = 0
    last_end = 0
    result_contents = []
    for edit, start_offset, end_offset in all_edits:
        result_contents.append(content[last_end:start_offset])
        mapping.append((last_end, start_offset, change))
        result_contents.append(edit.new_text)
        change += len(edit.new_text) - (end_offset - start_offset)
        last_end = end_offset
    result_contents.append(content[last_end:])
    
    result_content = ''.join(result_contents) 
    
    mapping.append((last_end, len(content), change))
    new_line_sizes = [len(l) + 1 for l in result_content.split('\n')]
    
    def find_change(offset: int) -> int | None:
        for start, end, change in mapping:
            if start <= offset < end:
                return change
        return None
    
    new_point_offsets = []
    for p in point_offsets:
        change = find_change(p)
        assert change is not None
        new_p = p + change
        new_point_offsets.append(new_p)
    new_points = [offset_to_line_column(p, new_line_sizes) for p in new_point_offsets]
    old_points = [offset_to_line_column(p, line_sizes) for p in point_offsets]
    
    point_mapping = {old: new for old, new in zip(old_points, new_points)}
    result_points = [point_mapping[p] for p in points]
    return result_content, result_points

class MacroExpansionResult:
    def __init__(self, kind: Literal['ok', 'err', 'halt'], *, 
                 result: tuple[str, list[Point]] | None = None, 
                 error_messages: list[str] | None = None) -> None:
        self.kind = kind
        self.result = result
        self.error_message = error_messages
        match kind:
            case 'ok':
                assert result is not None
            case 'err':
                assert error_messages is not None

def __expand_macro(clangd: ClangD, file: str, points: list[Point]) -> MacroExpansionResult:
    with open(file, 'r') as f:
        content = f.read()
    try:
        tokens = contain_leak('collect_type_and_identifiers', content, points[0], points[1])
    except:
        print(f'Error for {file=}, {points=}')
        raise
    macros: list[tuple[str, Point, Point]] = []
    for token in tokens:
        assert isinstance(token, tuple)
        semantic_token = clangd.get_semantic_token(file, token[1])
        if semantic_token is None:
            continue
        kind = semantic_token['type']
        if kind == 'macro':
            macros.append(token)
    macros = random_permutation(macros)
    error_messages = []
    for _, macro_start, macro_end in macros:
        edits = clangd.get_macro_expansion(file, macro_start, macro_end)
        new_content, new_points = apply_edits(content, edits, points)
        if new_content == content:
            continue
        return MacroExpansionResult('ok', result=(new_content, list(new_points)))
    if error_messages:
        return MacroExpansionResult('err', error_messages=error_messages)
    return MacroExpansionResult('halt')

def expand_macro(clangd: ClangD, file: str, points: list[Point]) -> tuple[list[Point], list[str]]:
    error_messages = []
    while True:
        expand_result = __expand_macro(clangd, file, points)
        match expand_result.kind:
            case 'halt':
                break
            case 'err':
                assert expand_result.error_message is not None
                error_messages.extend(expand_result.error_message)
                break
            case 'ok':
                assert expand_result.result is not None
                new_content, points = expand_result.result
                clangd.refresh_file_content(file, new_content)
    return points, error_messages

def trace_non_macro(clangd: ClangD, file: str, start_point: Point, end_point: Point) -> tuple[list[CodeItem], list[str]]:
    error_messages = []
    with open(file, 'r') as f:
        content = f.read()
    try:
        tokens = contain_leak('collect_type_and_identifiers', content, start_point, end_point)
    except:
        print(f'Error for {file=}, {start_point=}, {end_point=}')
        raise
    identifiers: list[tuple[str, Point, Point]] = []
    result = []
    for token in tokens:
        assert isinstance(token, tuple)
        semantic_token = clangd.get_semantic_token(file, token[1])
        if semantic_token is None:
            continue
        kind = semantic_token['type']
        if kind != 'macro':
            identifiers.append(token)
    for identifier in identifiers:
        start_point, end_point = identifier[1], identifier[2]
        defs = clangd.get_definition(file, start_point[0], start_point[1])
        if not defs:
            error_messages.append(f'Failed to find definition for {identifier}, {file=}')
            continue
        if len(defs) > 1:
            error_messages.append(f'Multiple definitions for {identifier}, {file=}')
        def_ = defs[0]
        def_file = uri_to_path(def_['uri'])

        def_file_dir = os.path.abspath(os.path.dirname(def_file))
        if not def_file_dir.startswith(PROJECT_ROOT):
            code_item = CodeItem('include_file', def_file)
            result.append(code_item)
            continue

        ref_start_point = (def_['range']['start']['line'], def_['range']['start']['character'])
        ref_end_point = (def_['range']['end']['line'], def_['range']['end']['character'])
        code_item: CodeItem = contain_leak('get_code_item', def_file, ref_start_point, ref_end_point)
        if code_item.kind == 'macrodef':
            warning = f'Code item is a macro definition {code_item}, {file=}, {start_point=}, {end_point=}'
            error_messages.append(warning)
        result.append(code_item)
    return result, error_messages

class CodeItemWithOrder:
    def __init__(self, code_item: CodeItem, order: int):
        self.code_item = code_item
        self.order = order
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, CodeItemWithOrder):
            return False
        return self.code_item == value.code_item
    
    def __hash__(self) -> int:
        return hash(self.code_item)
    
    def toJson(self) -> dict:
        return {
            'code_item': self.code_item.toJson(),
            'order': self.order
        }

def _topo_sort(parents: dict[CodeItem, list[CodeItem]], all_items: list[CodeItem]) -> list[CodeItem]:
    def dfs(item: CodeItem, visited: set[CodeItem], result: list[CodeItem]):
        if item in visited:
            return
        visited.add(item)
        assert item in parents, f'{item=}'
        for parent in parents[item]:
            dfs(parent, visited, result)
        result.append(item)
    
    result = []
    visited = set()
    for item in all_items:
        dfs(item, visited, result)
    return result

def trace_func(clangd: ClangD, file: str, func_name: str, start_point: Point) -> tuple[list[CodeItemWithOrder], list[CodeItemWithOrder], list[CodeItemWithOrder], list[str]]:
    ast = aa.get_ast_of_func_exact_match(file, start_point, func_name)
    assert ast is not None
    
    func_collect = set()
    other_collect = set()
    include_collect = set()
    warnings = set()
    
    parents = {}

    root_item = CodeItem('funcdef', file, (start_point[0], start_point[1]), (ast.end_point.row, ast.end_point.column), name=func_name)
    visited = set()

    MAX_NUM = 100
    worklist = [root_item]
    while worklist:
        current = worklist.pop()
        if current in visited:
            continue
        visited.add(current)
        parent = parents.pop(current, None)
        if parent is not None:
            parents[current] = parent
        for message in messages:
            warnings.add(message)
        code_items, messages = trace_non_macro(clangd, current.file, current.start_point, current.end_point)
        for message in messages:
            warnings.add(message)
            
        for code_item in code_items:
            if code_item != root_item:
                if code_item not in parents:
                    parents[code_item] = set()
                if current != code_item:
                    parents[code_item].add(current)
            match code_item.kind:
                case 'funcdef':
                    func_collect.add(code_item)
                    continue
                case 'include_file':
                    include_collect.add(code_item)
                    continue
                case _:
                    other_collect.add(code_item)
            if len(func_collect) + len(include_collect) + len(other_collect) > MAX_NUM:
                warnings.add(f'Exceed maximum number of items {MAX_NUM}')
                break
            for code_item in code_items:
                match code_item.kind:
                    case 'funcdef':
                        func_collect.add(code_item)
                        continue
                    case 'include_file':
                        include_collect.add(code_item)
                        continue
                    case _:
                        other_collect.add(code_item)
                worklist.append(code_item)
    assert root_item not in parents
    parents[root_item] = set()
    for item, the_parents in parents.items():
        if item != root_item:
            assert the_parents
    
    func_collect_list = list(func_collect)
    include_collect_list = list(include_collect)
    other_collect_list = list(other_collect)
    all_items = func_collect_list + include_collect_list + other_collect_list
    sorted_items = _topo_sort(parents, all_items)
    indices = {item: i for i, item in enumerate(sorted_items)}
    func_collect_with_order = [CodeItemWithOrder(item, indices[item]) for item in func_collect_list]
    include_collect_with_order = [CodeItemWithOrder(item, indices[item]) for item in include_collect_list]
    other_collect_with_order = [CodeItemWithOrder(item, indices[item]) for item in other_collect_list]
    return func_collect_with_order, include_collect_with_order, other_collect_with_order, list(warnings)

import click
import json
from tqdm import tqdm

BATCH_SIZE = 100

@click.command()
@click.option('--src', '-i', type=click.Path(exists=True, dir_okay=True, file_okay=False), help='Path to the source file')
@click.option('--func-list', '-l', type=click.Path(exists=True, dir_okay=False, file_okay=True), help='Path to the function list')
@click.option('--output', '-o', type=click.Path(exists=False, dir_okay=False, file_okay=True), 
              help='Output file for the extracted code', default='func_depends_%r.jsonl')
@click.option('--start-batch', '-s', type=int, default=-1, help='Start batch')
@click.option('--end-batch', '-e', type=int, default=-1, help='End batch')
@click.option('--batch-size', '-b', type=int, default=BATCH_SIZE, help='Batch size')
def main(src, func_list, output, start_batch, end_batch, batch_size):
    all_func_list = json.load(open(func_list, 'r'))

    batch_num = len(all_func_list) // batch_size
    if start_batch == -1 or end_batch == -1:
        print(f'There are {batch_num} batches. Please specify the start and end batch.')
        return
    if start_batch < 0 or start_batch >= batch_num or end_batch < 0 or end_batch >= batch_num:
        print(f'Invalid start_batch or end_batch')
        return
    batches = [all_func_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
    if len(all_func_list) % batch_size != 0:
        batches[-1].extend(all_func_list[batch_num*batch_size:])
    
    clangd = ClangD(src, src)
    working_batches = batches[start_batch:end_batch+1]
    all_num = sum(len(batch) for batch in working_batches)
    print(f'{start_batch}-{end_batch} / {batch_num - 1}')
    with tqdm(total=all_num) as pbar:
        for i, batch in enumerate(batches):
            output_file = output.replace('%r', str(i))
            pbar.desc = f'Batch {i} / {batch_num - 1}'
            with open(output_file, 'a') as f:
                for func in batch:
                    file = os.path.join(src, func['file'])
                    start_point = Point(func['start_point'])
                    func_name = func['name']
                    func_depends, include_depends, other_depends, warnings = trace_func(clangd, file, func_name, start_point)
                    item = {
                        'func': func,
                        'func_depends': [f.toJson() for f in func_depends],
                        'other_depends': [f.toJson() for f in other_depends],
                        'include_depends': [f.toJson() for f in include_depends],
                        'warnings': warnings
                    }
                    f.write(json.dumps(item) + '\n')
                    pbar.update(1)

if __name__ == '__main__':
    main()