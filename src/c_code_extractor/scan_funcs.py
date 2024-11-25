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
def main(src, output):
    all_files = []
    for p, ds, fs in os.walk(src):
        for f in fs:
            if f.endswith('.c'):
                all_files.append(os.path.join(p, f))
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
            relative_item = CodeItem(func.kind, func.name, func.file.removeprefix(src + '/'), (start_point[0], start_point[1]), (end_point[0], end_point[1]))
            results.append(relative_item.toJson())
    print(f'{count}/{len(results)}')
    with open(output, 'w') as f:
        f.write(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()