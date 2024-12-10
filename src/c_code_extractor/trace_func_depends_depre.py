import os
import os.path
import click
import json
import ast_analyze as aa
from ast_analyze import Point
from lsp_client import ClangD, Edit
from lsp_server import uri_to_path
from code_item import CodeItem
from tqdm import tqdm
from defs import PROJECT_ROOT
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence
from datetime import datetime
from util import *

RangeMapping = dict[tuple[int, int], int]

def apply_edits(context: str, edits: Sequence[Edit]) -> tuple[str, RangeMapping, RangeMapping]:
    lines = context.split('\n')
    
    line_sizes = [len(line) + 1 for line in lines]
    
    edits_with_offset = []
    for edit in edits:
        start_row, start_column = edit.start_point
        start_offset = line_column_to_offset(start_row, start_column, line_sizes)
        end_row, end_column = edit.end_point
        end_offset = line_column_to_offset(end_row, end_column, line_sizes)
        edits_with_offset.append((start_offset, end_offset, edit.new_text))

    edits_with_offset.sort(key=lambda x: x[0])
    to_pop = []
    for i in range(1, len(edits_with_offset)):
        if not edits_with_offset[i][0] >= edits_with_offset[i-1][1]:
            print(f'Overlapping edits {edits_with_offset[i-1]=}, {edits_with_offset[i]=}')
            to_pop.append(i)
    for i in reversed(to_pop):
        edits_with_offset.pop(i)
    
    reverse_mapping: RangeMapping = {(0, edits_with_offset[0][0]): 0}
    mapping: RangeMapping = {(0, edits_with_offset[0][0]): 0} 
    size_change = 0
    snippets = []
    last_end_offset = 0
    for i, (start_offset, end_offset, new_text) in enumerate(edits_with_offset):
        
        range_l = last_end_offset + size_change
        range_r = start_offset + size_change
        reverse_mapping[(range_l, range_r)] = -size_change
        mapping[(last_end_offset, start_offset)] = size_change
        
        size_change += len(new_text) - (end_offset - start_offset)
        snippets.append(context[last_end_offset:start_offset])
        snippets.append(new_text)
        last_end_offset = end_offset
    range_l = last_end_offset + size_change
    range_r = len(context) + size_change
    reverse_mapping[(range_l, range_r)] = -size_change
    mapping[(last_end_offset, len(context))] = size_change
    
    snippets.append(context[last_end_offset:])
    new_contents = ''.join(snippets)
    return new_contents, reverse_mapping, mapping

def map_range(content: str, new_content: str, mapping: dict[tuple[int, int], int], point: Point) -> Point | str:
    line_sizes = [len(line) + 1 for line in content.split('\n')]
    new_line_sizes = [len(line) + 1 for line in new_content.split('\n')]
    
    offset = line_column_to_offset(point[0], point[1], line_sizes)
    items = list(sorted(mapping.items(), key=lambda x: x[0][0]))
    for i, ((l, r), size_change) in enumerate(items):
        if l <= offset < r:
            retval = Point(offset_to_line_column(offset + size_change, new_line_sizes))
            return retval
        if i >= 1 and items[i-1][0][1] <= offset < l:
            retval = Point(offset_to_line_column(items[i-1][0][1], new_line_sizes))
            return retval
    return f'{point=}, {offset=}, {mapping=}'

def apply_edits_multiworld(context: str, edit_batches: Sequence[Sequence[Edit]]) -> Sequence[tuple[str, RangeMapping, RangeMapping]]:
    worlds = []
    for edits in edit_batches:
        new_content, mapping_back, mapping = apply_edits(context, edits)
        worlds.append((new_content, mapping_back, mapping))
    return worlds

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

def extract_func(clangd: ClangD, file: str, start_point: Point, func_name: str) -> tuple[list[CodeItemWithOrder], list[CodeItemWithOrder], list[CodeItemWithOrder], list[str]]:
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
    work_list = [(root_item, (file, start_point, (ast.end_point.row, ast.end_point.column)))]
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as executor:
        while work_list:
            item, current = work_list.pop()
            
            if item in visited or item != root_item and aa.is_within((item.start_point, item.end_point), (root_item.start_point, root_item.end_point)):
                continue
            if len(other_collect) + len(func_collect) + len(include_collect) > MAX_NUM:
                warning = f'Warning: Too many items: {len(other_collect) + len(func_collect) + len(include_collect)}'
                print(warning)
                warnings.add(warning)
                break
            
            visited.add(item)

            if item != root_item:
                other_collect.add(item)
            
            with open(item.file, 'r') as f:
                old_content = f.read()
            item_semantic_token = clangd.get_semantic_token(item.file, item.start_point)
            if item_semantic_token is not None and item_semantic_token['type'] == 'comment':
                f = executor.submit(aa.cancel_macro, old_content, item.start_point, item.end_point)
                refreshed_content = f.result()
                clangd.refresh_file_content(item.file, refreshed_content)
            else:
                refreshed_content = old_content

            def one_round(current_ast_loc, *, 
                        macro_mode=True,
                        old_content: str = '',
                        new_content: str = '', 
                        mapping_back: RangeMapping = {}) -> Sequence[tuple[str, Point, Point]]:
                # tokens = collect_type_and_identifiers(current_ast)
                f = executor.submit(aa.leak_wrapper, 'collect_type_and_identifiers', current_ast_loc)
                tokens = f.result()
                macros: list[tuple[str, Point, Point]] = []
                for token in tokens:
                    semantic_token = clangd.get_semantic_token(item.file, token[1])
                    if semantic_token is None:
                        continue
                    kind = semantic_token['type']
                    if kind == 'macro':
                        macros.append(token)
            
                within_macros = set()
                for macro in macros:
                    try:
                        f = executor.submit(aa.leak_wrapper, 'get_macro_expanding_range', current_ast_loc, macro[1], macro[2])
                        range_ = f.result()
                    except:
                        print(f'Error when processing {item=}, {macro=}')
                        raise
                    within_macros.add(range_)

                def within_macro(start_point, end_point) -> bool:
                    for range_ in within_macros:
                        if aa.is_within((start_point, end_point), range_):
                            return True
                    return False

                for i, (id_name, start_point, end_point) in enumerate(tokens):
                    # print(f'One-round token {i + 1} / {len(tokens)}')
                    if macro_mode:
                        if not within_macro(start_point, end_point):
                            continue
                    else:
                        if within_macro(start_point, end_point):
                            continue
                    
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
                    # code_item = get_code_item(def_file, ref_start_point, ref_end_point)
                    f = executor.submit(aa.get_code_item, def_file, ref_start_point, ref_end_point)
                    code_item = f.result()
                    
                    if code_item is None:
                        warning = f'No code item: {item=}, {id_name=}, {start_point=}, {end_point=}, {def_file=}, {ref_start_point=}, {ref_end_point=}'
                        warnings.add(warning)
                        continue

                    if code_item.kind == 'funcdef':
                        if item.file == code_item.file and not macro_mode:
                            real_start_point = map_range(new_content, old_content, mapping_back, code_item.start_point)
                            real_end_point = map_range(new_content, old_content, mapping_back, code_item.end_point) 
                            assert not isinstance(real_start_point, str)
                            assert not isinstance(real_end_point, str)
                            code_item = CodeItem(code_item.kind, code_item.file, real_start_point, real_end_point, name=code_item.name)
                    
                        if code_item != root_item and code_item not in parents:
                            parents[code_item] = set()
                        if code_item != root_item:
                            parents[code_item].add(item)

                            func_collect.add(code_item)
                    else:
                        ast_loc = (def_file, code_item.start_point, code_item.end_point)
                        # assert ast is not None
                        if item.file == code_item.file and not macro_mode:
                            real_start_point = map_range(new_content, old_content, mapping_back, code_item.start_point)
                            real_end_point = map_range(new_content, old_content, mapping_back, code_item.end_point) 
                            assert not isinstance(real_start_point, str)
                            assert not isinstance(real_end_point, str)
                            code_item = CodeItem(code_item.kind, code_item.file, real_start_point, real_end_point, name=code_item.name)
                        if code_item not in parents:
                            parents[code_item] = set()
                        parents[code_item].add(item)
                        work_list.append((code_item, ast_loc))
                return macros if macro_mode else []
            
            macros = list(one_round(current))
            original_num = len(macros)
            
            existing = set()
            to_pop = []
            for i, (macro_name, macro_start, macro_end) in enumerate(macros):
                if macro_name in existing:
                    to_pop.append(i)
                else:
                    existing.add(macro_name)
            for i in reversed(to_pop):
                macros.pop(i)
            
            print(f'Macro num: {original_num} -> {len(macros)}')
            changes = []
            for _, macro_start, macro_end in macros:
                tmp = clangd.get_macro_expansion(item.file, macro_start, macro_end)
                if not tmp:
                    continue
                changes.append(tmp)

            multiworlds = apply_edits_multiworld(refreshed_content, changes)

            start = datetime.now()
            remaining = -1
            elapsed = 0
            MAX_TIME = 60 * 20
            for i, (new_content, mapping_back, mapping) in enumerate(multiworlds):
                if elapsed > MAX_TIME:
                    warning = f'World process timeout: {item=}, {len(multiworlds)=}, {i=}.'
                    warnings.add(warning)
                    break
                print(f'({elapsed // 60} / {remaining // 60 if remaining > 0 else remaining}) World {i + 1} / {len(multiworlds)}')
                clangd.refresh_file_content(item.file, new_content)
                new_start_point = map_range(refreshed_content, new_content, mapping, item.start_point)
                new_end_point = map_range(refreshed_content, new_content, mapping, item.end_point)
                
                mapping_error = False
                if isinstance(new_start_point, str):
                    warning = f'Error when mapping range: {item=}, with message {new_start_point}'
                    warnings.add(warning)
                    mapping_error = True
                if isinstance(new_end_point, str):
                    warning = f'Error when mapping range: {item=}, with message {new_end_point}'
                    warnings.add(warning)
                    mapping_error = True
                if mapping_error:
                    continue
                
                assert not isinstance(new_start_point, str)
                assert not isinstance(new_end_point, str)
                current_ast_loc = (item.file, new_start_point, new_end_point)
                try:
                    one_round(current_ast_loc, macro_mode=False, old_content=refreshed_content, 
                            new_content=new_content, mapping_back=mapping_back)
                except aa.NoASTError:
                    warning = f'Error when getting AST: {item=}, {new_start_point=}, {new_end_point=}, new_content=|{new_content}|'
                    warnings.add(warning)
                    continue
                elapsed = (datetime.now() - start).seconds
                average = elapsed / (i + 1)
                remaining = min(MAX_TIME, average * (len(multiworlds) - i))
            clangd.refresh_file_content(item.file, old_content)
        
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
        for j, func in enumerate(batches[i]):
            print(f'{tqdm_tag}: func {j + 1} / {len(batches[i])}')
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
@click.option('--split/--no-split', '-p/-m', help='Split the output file')
@click.option('--batch-size', '-b', type=int, default=BATCH_SIZE, help='Batch size')
def main(src, func_list, output, start_batch, end_batch, split, batch_size):
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
            start = datetime.now()
            elapsed = 0
            remaining = -1
            for i in range(start_batch, end_batch + 1):
                batch_count = i - start_batch + 1
                f = executor.submit(process_one_batch, i, output, batches, src, 
                                    tqdm_tag=f'({elapsed // 60} / {remaining // 60}) Batch {i} [{batch_count} / {all_bacthes}]')
                f.result()
                elapsed = (datetime.now() - start).seconds
                
                average = elapsed / (i - start_batch + 1)
                remaining = average * (end_batch - i)
            
        
if __name__ == '__main__':
    main()