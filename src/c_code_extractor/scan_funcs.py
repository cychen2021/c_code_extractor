import click
import json
import os
import os.path
from ast_analyze import *
from tqdm import tqdm

LINE_NUM_THREASHOLD=100

@click.command()
@click.option('--src', '-i', type=click.Path(exists=True, dir_okay=True, file_okay=False), help='Path to the source file')
@click.option('--output', '-o', type=click.Path(exists=False, dir_okay=False, file_okay=True), 
              help='Output file for the extracted code', default='callgraph.json')
@click.option('--compile-commands', '-c', type=click.Path(exists=True, dir_okay=True, file_okay=False), default=None)
@click.option('--exclude-header/--include-header', '-e/-I', is_flag=True, default=True)
def main(src, output, exclude_header, compile_commands):
    if compile_commands is None:
        compile_commands = os.path.join(src, 'compile_commands.json')
    
    with open(compile_commands, 'r') as f:
        compile_commands = json.load(f)
    
    all_files = []
    for command in compile_commands:
        file_path = command['file']
        if file_path.endswith('.c') or (not exclude_header and file_path.endswith('.h')):
            all_files.append(file_path)
    results = []
    count = 0
    for f in tqdm(all_files):
        funcs = get_all_funcs(f)
        for func in funcs:
            start_point = func.start_point
            end_point = func.end_point
            line_num = end_point[0] - start_point[0] + 1
            if line_num < LINE_NUM_THREASHOLD:
                count += 1
            relative_item = CodeItem(func.kind, func.file.removeprefix(src + '/'), (start_point[0], start_point[1]), 
                                     (end_point[0], end_point[1]), name=func.name)
            results.append(relative_item.to_json())
    print(f'{count}/{len(results)} < {LINE_NUM_THREASHOLD} lines')
    with open(output, 'w') as f:
        f.write(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()