import click
from lsp_server import *
from lsp_client import *

@click.command('crawl')
@click.option('--src', type=click.Path(exists=True, dir_okay=True, file_okay=False), help='Path to the source file')
@click.option('--build', type=click.Path(exists=True, dir_okay=True, file_okay=False), help='Path to the build directory', default=None)
@click.option('--entry', type=str, help='Entry point for the code extraction')
def crawl(src, build, entry):
    if build is None:
        build = src
    
    clangd = ClangD(src, build)
    
    entry_file, entry_function = entry.split(':')
    entry_func_info = clangd.get_function_by_name(entry_function, entry_file)
    print(entry_func_info)
    
if __name__ == '__main__':
    crawl()
