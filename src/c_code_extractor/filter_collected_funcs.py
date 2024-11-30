import click
import json
import os
import os.path

@click.command()
@click.option('--compilation-db', '-db', type=click.Path(exists=True, dir_okay=False, file_okay=True), help='Path to the compilation database')
@click.option('--src-prefix', '-p', type=click.Path(exists=True, dir_okay=True, file_okay=False), help='Prefix of the source files', default=None)
@click.argument('func_list', type=click.Path(exists=True, dir_okay=False, file_okay=True))
def main(compilation_db, func_list, src_prefix):
    with open(compilation_db, 'r') as f:
        compilation_commands = json.load(f)
    if src_prefix is None:
        src_prefix = os.path.abspath(os.path.dirname(compilation_db)) + '/'
    files = [command['file'].replace(r'//', '/').removeprefix(src_prefix) for command in compilation_commands]
    with open(func_list, 'r') as f:
        funcs = json.load(f)
    original_num = len(funcs)
    filtered = []
    for func in funcs:
        if func['file'] in files:
            filtered.append(func)
    after_num = len(filtered)
    print(f'{original_num} -> {after_num}')
    with open(func_list, 'w') as f:
        f.write(json.dumps(filtered, indent=2))
        
if __name__ == '__main__':
    main()
