from ast_analyze import Point
from trace_func_depends import expand_macro
from lsp_client import ClangD
import click
import os.path
import json
from tqdm import tqdm


def expand_all(clangd: ClangD, file: str, points: list[Point]) -> tuple[list[Point], list[str]]:
    points, error_messages = expand_macro(clangd, file, points)
    return points, error_messages

BATCH_SIZE = 10

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
    
    file_map = {}
    for func in all_func_list:
        file = os.path.join(src, func['file'])
        if file not in file_map:
            file_map[file] = []
        file_map[file].append(func)
    
    acc_warnings = {}
    
    clangd = ClangD(src, src)
    working_batches = batches[start_batch:end_batch+1]
    all_num = sum(len(batch) for batch in working_batches)
    print(f'{start_batch}-{end_batch} / {batch_num - 1}')
    with tqdm(total=all_num) as pbar:
        for i, batch in enumerate(batches):
            pbar.desc = f'Batch {i} / {batch_num - 1}'
            for func in batch:
                file = os.path.join(src, func['file'])
                
                start_point = Point(func['start_point'])
                end_point = Point(func['end_point'])
                
                points = [start_point, end_point]
                funcs = [func]
                for another_func in file_map[file]:
                    if another_func != func:
                        points.append(Point(another_func['start_point']))
                        points.append(Point(another_func['end_point']))
                        funcs.append(another_func)
                points, warnings = expand_all(clangd, file, points)

                for i in range(len(funcs)):
                    funcs[i]['start_point'] = points[2*i]
                    funcs[i]['end_point'] = points[2*i+1]
                
                assert isinstance(func['file'], str)
                assert isinstance(func['name'], str)
                acc_warnings[(func['file'], func['name'])] = warnings
                pbar.update(1)
    result = []
    for _, funcs in file_map.items():
        for func in funcs:
            item = {
                **func,
                'warnings': acc_warnings[(func['file'], func['name'])]
            }
            result.append(item)
    with open(output, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    main()