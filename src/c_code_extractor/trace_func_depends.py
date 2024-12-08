import tree_sitter as ts
import tree_sitter_c as tsc
import os
import os.path
import click
import json
from ast_analyze import *
from lsp_client import ClangD, Edit
from lsp_server import uri_to_path
from code_item import CodeItem
from tqdm import tqdm
from defs import PROJECT_ROOT
from concurrent.futures import ProcessPoolExecutor
import tempfile
from typing import Sequence

def line_column_to_offset(line: int, column: int, line_sizes: Sequence[int]) -> int:
    return sum(line_sizes[:line]) + column

def offset_to_line_column(offset: int, line_sizes: Sequence[int]) -> tuple[int, int]:
    line = 0
    while line < len(line_sizes) and offset >= line_sizes[line]:
        offset -= line_sizes[line]
        line += 1
    return line, offset

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

def map_range(content: str, new_content: str, mapping: dict[tuple[int, int], int], point: Point) -> Point:
    line_sizes = [len(line) + 1 for line in content.split('\n')]
    new_line_sizes = [len(line) + 1 for line in new_content.split('\n')]
    
    offset = line_column_to_offset(point[0], point[1], line_sizes)
    for (l, r), size_change in mapping.items():
        if l <= offset < r:
            retval = Point(offset_to_line_column(offset + size_change, new_line_sizes))
            return retval
    assert False, f'{point=}, {offset=}, {mapping=}'

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
        
        with open(item.file, 'r') as f:
            old_content = f.read()
        item_semantic_token = clangd.get_semantic_token(item.file, item.start_point)
        if item_semantic_token is not None and item_semantic_token['type'] == 'comment':
            to_cancel = cancel_macro(old_content, item.start_point, item.end_point)
            if to_cancel is not None:
                mask, to_erase = to_cancel
                lines = old_content.split('\n')
                for row in range(to_erase[0][0], to_erase[1][0] + 1):
                    for column in range(0 if row != to_erase[0][0] else to_erase[0][1], 
                                        len(lines[row]) if row != to_erase[1][0] else to_erase[1][1]):
                        if not point_is_within((row, column), mask):
                            lines[row] = lines[row][:column] + ' ' + lines[row][column+1:]
                new_content = '\n'.join(lines)
                with tempfile.NamedTemporaryFile('w', delete=False) as f:
                    new_file = f.name
                    f.write(new_content)
                current = get_ast_exact_match(new_file, item.start_point, item.end_point)
                assert current is not None
                os.remove(new_file)
                clangd.refresh_file_content(item.file, new_content)
                refreshed_content = new_content
        else:
            refreshed_content = old_content

        def one_round(current_ast, *, 
                      macro_mode=True,
                      old_content: str = '',
                      mapping: RangeMapping = {},
                      new_content: str = '', 
                      mapping_back: RangeMapping = {}) -> Sequence[tuple[str, Point, Point]]:
            tokens = collect_type_and_identifiers(current_ast)
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
                range_ = get_macro_expanding_range(current_ast, macro[1], macro[2])
                within_macros.add(range_)

            def within_macro(start_point, end_point) -> bool:
                for range_ in within_macros:
                    if is_within((start_point, end_point), range_):
                        return True
                return False

            for id_name, start_point, end_point in tokens:
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

                        if item.file == code_item.file and not macro_mode:
                            real_start_point = map_range(new_content, old_content, mapping_back, code_item.start_point)
                            real_end_point = map_range(new_content, old_content, mapping_back, code_item.end_point) 
                            code_item = CodeItem(code_item.kind, code_item.file, real_start_point, real_end_point, name=code_item.name)
                
                        func_collect.add(code_item)
                else:
                    ast = get_ast_exact_match(def_file, code_item.start_point, code_item.end_point)
                    assert ast is not None
                    if item.file == code_item.file and not macro_mode:
                        real_start_point = map_range(new_content, old_content, mapping_back, code_item.start_point)
                        real_end_point = map_range(new_content, old_content, mapping_back, code_item.end_point) 
                        code_item = CodeItem(code_item.kind, code_item.file, real_start_point, real_end_point, name=code_item.name)
                    if code_item not in parents:
                        parents[code_item] = set()
                    parents[code_item].add(item)
                    work_list.append((code_item, ast))
            return macros if macro_mode else []
        
        macros = one_round(current)
        
        changes = []
        for _, macro_start, macro_end in macros:
            tmp = clangd.get_macro_expansion(item.file, macro_start, macro_end)
            if not tmp:
                continue
            changes.append(tmp)

        multiworlds = apply_edits_multiworld(refreshed_content, changes)
        for new_content, mapping_back, mapping in multiworlds:
            clangd.refresh_file_content(item.file, new_content)
            new_start_point = map_range(refreshed_content, new_content, mapping, item.start_point)
            new_end_point = map_range(refreshed_content, new_content, mapping, item.end_point)
            current = get_ast_exact_match(item.file, new_start_point, new_end_point)
            assert current is not None, f'{item.file=} {new_start_point=}, {new_end_point=}'
            one_round(current, macro_mode=False, old_content=refreshed_content, mapping=mapping, 
                      new_content=new_content, mapping_back=mapping_back)
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
@click.option('--split/--no-split', '-p/-m', help='Split the output file')
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
                if func['name'] != 'ABRThandler':
                    continue
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
        from datetime import datetime
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
                
                average = elapsed / (end_batch - start_batch + 1)
                remaining = average * (end_batch - i)
            
        
if __name__ == '__main__':
    main()