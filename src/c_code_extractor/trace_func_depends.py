import tree_sitter as ts
import tree_sitter_c as tsc
import os
import os.path
import click
import json
from ast_analyze import *
from lsp_client import ClangD
from lsp_server import uri_to_path
from code_item import CodeItem
from tqdm import tqdm
from defs import PROJECT_ROOT
from concurrent.futures import ProcessPoolExecutor

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
        for parent in parents[item]:
            dfs(parent, visited, result)
        result.append(item)
    
    result = []
    visited = set()
    for item in all_items:
        dfs(item, visited, result)
    return result

def extract_func(clangd: ClangD, file: str, start_point: Point, func_name: str) -> tuple[list[CodeItemWithOrder], list[CodeItemWithOrder], list[CodeItemWithOrder], list[str]]:
    ast = get_ast_of_func_exact_match(file, start_point, func_name)
    assert ast is not None
    
    func_collect = set()
    other_collect = set()
    include_collect = set()
    warnings = set()
    
    parents = {}

    root_item = CodeItem('funcdef', file, (start_point[0], start_point[1]), (ast.end_point.row, ast.end_point.column), name=func_name)
    visited = set()

    MAX_NUM = 100
    work_list = [(root_item, ast)]
    while work_list:
        item, current = work_list.pop()
        
        if item in visited or item != root_item and is_within((item.start_point, item.end_point), (root_item.start_point, root_item.end_point)):
            continue

        if len(other_collect) + len(func_collect) + len(include_collect) > MAX_NUM:
            warning = f'Warning: Too many items: {len(other_collect) + len(func_collect) + len(include_collect)}'
            print(warning)
            warnings.add(warning)
            break
        
        visited.add(item)

        if item != root_item:
            other_collect.add(item)

        normal_identifiers = collect_identifiers(current)
        ty_identifiers = collect_types(current)
        identifiers = normal_identifiers + ty_identifiers
        
        for id_name, start_point, end_point in identifiers:
            defs = clangd.get_definition(item.file, start_point[0], start_point[1])
            if len(defs) == 0:
                continue
            elif len(defs) > 1:
                warning = f'Multiple definitions: {item=}, {id_name=}, {start_point=}, {end_point=}'
                warnings.add(warning)
                continue
            def_ = defs[0]
            def_file = uri_to_path(def_['uri'])

            def_file_dir = os.path.abspath(os.path.dirname(def_file))
            if not def_file_dir.startswith(PROJECT_ROOT):
                code_item = CodeItem('include_file', def_file)
                if code_item not in parents:
                    parents[code_item] = set()
                    
                parents[code_item].add(item)
                include_collect.add(code_item)
                continue
            
            ref_start_point = (def_['range']['start']['line'], def_['range']['start']['character'])
            ref_end_point = (def_['range']['end']['line'], def_['range']['end']['character'])
            code_item = get_code_item(def_file, ref_start_point, ref_end_point)
            if code_item is None:
                warning = f'No code item: {item=}, {id_name=}, {start_point=}, {end_point=}, {def_file=}, {ref_start_point=}, {ref_end_point=}'
                warnings.add(warning)
                continue
            if code_item.kind == 'funcdef':
                if code_item != root_item and code_item not in parents:
                    parents[code_item] = set()
                if code_item != root_item:
                    parents[code_item].add(item)
                    func_collect.add(code_item)
            else:
                ast = get_ast_exact_match(def_file, code_item.start_point, code_item.end_point)
                assert ast is not None
                if code_item not in parents:
                    parents[code_item] = set()
                parents[code_item].add(item)
                work_list.append((code_item, ast))
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

def process_one_batch(i, output, batches, src, tqdm_tag: str | None = None):
    clangd = ClangD(src, src)
    with open(output.replace('%r', f'{i}'), 'w') as f:
        for func in tqdm(batches[i], desc=tqdm_tag):
            func_depends, include_depends, other_depends, warnings = extract_func(clangd, os.path.join(src, func['file']), func['start_point'], func['name'])
            item = {
                'func': func,
                'func_depends': [f.toJson() for f in func_depends],
                'other_depends': [f.toJson() for f in other_depends],
                'include_depends': [f.toJson() for f in include_depends],
                'warnings': warnings
            }
            f.write(json.dumps(item) + '\n')
            del item, func_depends, include_depends, other_depends, warnings

BATCH_SIZE = 10
    
@click.command()
@click.option('--src', '-i', type=click.Path(exists=True, dir_okay=True, file_okay=False), help='Path to the source file')
@click.option('--func-list', '-l', type=click.Path(exists=True, dir_okay=False, file_okay=True), help='Path to the function list')
@click.option('--output', '-o', type=click.Path(exists=False, dir_okay=False, file_okay=True), 
              help='Output file for the extracted code', default='func_depends_%r.jsonl')
@click.option('--start-batch', '-s', type=int, default=-1, help='Start batch')
@click.option('--end-batch', '-e', type=int, default=-1, help='End batch')
@click.option('--split/--no-split', '-p/-m', is_flag=True, help='Split the output file', default=True)
def main(src, func_list, output, start_batch, end_batch, split):
    all_func_list = json.load(open(func_list, 'r'))

    batch_num = len(all_func_list) // BATCH_SIZE
    if start_batch == -1 or end_batch == -1:
        print(f'There are {batch_num} batches. Please specify the start and end batch.')
        return
    if start_batch < 0 or start_batch >= batch_num or end_batch < 0 or end_batch >= batch_num:
        print(f'Invalid start_batch or end_batch')
        return
    batches = [all_func_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for i in range(batch_num)]
    if len(all_func_list) % BATCH_SIZE != 0:
        batches[-1].extend(all_func_list[batch_num*BATCH_SIZE:])
    
    if not split:
        clangd = ClangD(src, src)
        func_list = []
        for i in range(start_batch, end_batch + 1):
            func_list.extend(batches[i])
        with open(output.replace('%r', f'{start_batch}_{end_batch}'), 'w') as f:
            for func in tqdm(func_list):
                func_depends, include_depends, other_depends, warnings = extract_func(clangd, os.path.join(src, func['file']), func['start_point'], func['name'])
                item = {
                    'func': func,
                    'func_depends': [f.toJson() for f in func_depends],
                    'other_depends': [f.toJson() for f in other_depends],
                    'include_depends': [f.toJson() for f in include_depends],
                    'warnings': warnings
                }
                f.write(json.dumps(item) + '\n')
                del item, func_depends, include_depends, other_depends, warnings
    else:
        # The code causes memory leak. Use a process to release the memory.
        with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as executor:
            all_bacthes = end_batch - start_batch + 1
            for i in range(start_batch, end_batch + 1):
                batch_count = i - start_batch + 1
                f = executor.submit(process_one_batch, i, output, batches, src, tqdm_tag=f'Batch {i} [{batch_count} / {all_bacthes}]')
                f.result()
            
        
if __name__ == '__main__':
    main()