from ast_analyze import Point
from trace_func_depends import expand_macro_ultimate
from lsp_client import ClangD
import click
import os.path
import json
from tqdm import tqdm
import copy


def expand_all(clangd: ClangD, file: str) -> list[str]:
    try:
        error_messages = expand_macro_ultimate(clangd, file)
    except:
        raise
    return error_messages

BATCH_SIZE = 10

@click.command()
@click.option('--src', '-i', type=click.Path(exists=True, dir_okay=True, file_okay=False), help='Path to the source file')
@click.option('--compile-commands', '-c', type=click.Path(exists=True, dir_okay=False, file_okay=True), default=None, help='Path to the compile_commands.json')
@click.option('--output', '-o', type=click.Path(exists=False, dir_okay=False, file_okay=True), 
              help='Output file for the extracted code', default='func_depends_%r.jsonl')
@click.option('--start-batch', '-s', type=int, default=-1, help='Start batch')
@click.option('--end-batch', '-e', type=int, default=-1, help='End batch')
@click.option('--batch-size', '-b', type=int, default=BATCH_SIZE, help='Batch size')
def main(src, output, start_batch, end_batch, batch_size, compile_commands):
    if compile_commands is None:
        compile_commands = os.path.join(src, 'compile_commands.json')
    
    all_files = []
    with open(compile_commands, 'r') as f:
        compile_commands = json.load(f)
        for item in compile_commands:
            if item['file'].endswith('.c') or item['file'].endswith('.h'):
                all_files.append(item['file'])

    batch_num = len(all_files) // batch_size
    if start_batch == -1 or end_batch == -1:
        print(f'There are {batch_num} batches. Please specify the start and end batch.')
        return
    if start_batch < 0 or start_batch >= batch_num or end_batch < 0 or end_batch >= batch_num:
        print(f'Invalid start_batch or end_batch')
        return
    
    batches = [all_files[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
    if len(all_files) % batch_size != 0:
        batches[-1].extend(all_files[batch_num*batch_size:])
    
    clangd = ClangD(src, src)
    working_batches = batches[start_batch:end_batch+1]
    all_num = sum(len(batch) for batch in working_batches)
    print(f'{start_batch}-{end_batch} / {batch_num - 1}')
    with tqdm(total=all_num) as pbar:
        for i, batch in enumerate(batches):
            with open(output.replace('%r', str(i)), 'w') as f:
                pbar.desc = f'Batch {i + start_batch} / {batch_num - 1}'
                for file in batch:
                    warnings = expand_all(clangd, file)
                    f.write(json.dumps({'file': file, 'warnings': warnings}) + '\n')

                    pbar.update(1)

if __name__ == '__main__':
    main()