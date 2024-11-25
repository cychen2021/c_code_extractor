import click
from lsp_server import *
from lsp_client import *
from ast_analyze import *
from networkx import DiGraph
from typing import Any
from rw_lock import RWLock
import dill
from readchar import readchar
import threading
from base64 import b64encode, b64decode
import os

class CrawlResults:
    def __init__(self, entry_function: str, entry_file: str, 
                 graph: DiGraph, explored: int) -> None:
        self.graph = graph
        self.lock = RWLock()
        self.entry_function = entry_function
        self.entry_file = entry_file
        self.explored = explored
    
    def update_graph(self, graph: DiGraph):
        with self.lock.w_locked():
            self.graph = graph
    
    def update_explored(self, explored: int):
        with self.lock.w_locked():
            self.explored = explored

    def serilizable(self):
        with self.lock.r_locked():
            return {
                'entry_function': self.entry_function,
                'entry_file': self.entry_file,
                'graph': b64encode(dill.dumps(self.graph)).decode(),
                'explored': self.explored
            }
    
    def dumps(self) -> str:
        return json.dumps(self.serilizable())

EXIT_KEY = 'q'

@click.command('crawl')
@click.option('--src', '-i', type=click.Path(exists=True, dir_okay=True, file_okay=False), help='Path to the source file')
@click.option('--build', '-B', type=click.Path(exists=True, dir_okay=True, file_okay=False), help='Path to the build directory', default=None)
@click.option('--entry', '-E', type=str, help='Entry point for the code extraction')
@click.option('--output', '-o', type=click.Path(exists=False, dir_okay=False, file_okay=True), 
              help='Output file for the extracted code', default='callgraph.json')
@click.option('--resume', '-R', type=click.Path(exists=True, dir_okay=False, file_okay=True), 
              help='Resume the previous crawl', default=None)
def crawl(src, build, entry, output, resume):
    if build is None:
        build = src
    
    clangd = ClangD(src, build)
    
    def key_listener():
        while readchar() != EXIT_KEY:
            pass
        with open(output, 'w') as f:
            f.write(results.dumps())
        os._exit(0)
    
    thread = threading.Thread(target=key_listener)
    thread.daemon = True
    thread.start()
    
    if resume is None:
        entry_file, entry_function = entry.split(':')
        entry_func_info = clangd.get_function_by_name(entry_function, entry_file)
        assert entry_func_info is not None
        ast = get_ast_of_func(entry_func_info.file, entry_func_info.start_point, entry_func_info.end_point)
        root_node = (entry_file, entry_function, ast, entry_func_info.start_point, entry_func_info.end_point)
        to_explore: list[tuple[int | None, Any]] = [(None, root_node)]
        graph = DiGraph()
        
        results = CrawlResults(entry_function, entry_file, graph, 0)
        
        collect = set()
        count = 0
    else:
        with open(resume, 'r') as f:
            data = json.load(f)
        entry_file = data['entry_file']
        entry_function = data['entry_function']
        count = data['explored']
        
        graph: DiGraph = dill.loads(b64decode(data['graph']))
        collect = set()
        to_explore: list[tuple[int | None, Any]] = []
        for nid, node in graph.nodes.items():
            ast = get_ast_of_func(node.file, node.start, node.end)
            if graph.out_degree(nid) == 0:
                parents: list[int] = graph.predecessors(nid)
                for p in parents:
                    to_explore.append((p, (node.data['file'], node.data['name'], ast, node.data['start'], node.data['end'])))
            else:
                collect.add((node.data['file'], node.data['name'], node.data['start'], node.data['end']))
    while to_explore:
        parent, (current_file, name, current_ast, current_start, current_end) = to_explore.pop()
        if (current_file, name, current_start, current_end) in collect:
            continue

        print(f'Exploring {count + 1}th item {name} in {current_file}')

        nid = count
        graph.add_node(node_for_adding=nid, data={'file': current_file, 'name': name, 'start': current_start, 'end': current_end})
        if parent is not None:
            graph.add_edge(parent, nid)
        count += 1
        results.update_graph(graph)
        results.update_explored(count)
        
        collect.add((current_file, name, current_start, current_end))
        calls = collect_calls(current_ast)
        assert isinstance(name, str)
        for item, start_point, end_point in calls:
            def_infos = clangd.get_definition(current_file, start_point[0], start_point[1])
            if not def_infos:
                continue
            for def_info in def_infos:
                file = uri_to_path(def_info['uri'])
                start_point = (def_info['range']['start']['line'], def_info['range']['start']['character'])
                end_point = (def_info['range']['end']['line'], def_info['range']['end']['character'])
                if file == current_file and is_within((start_point, end_point), (current_start, current_end)):
                    continue
                if file.startswith('/'):
                    continue
                ast = get_ast_of_func(os.path.join(src, file), start_point, end_point)
                if ast is None:
                    continue
                to_explore.append((
                    nid,
                    (file, item, ast, start_point, end_point)
                ))
    with open(output, 'w') as f:
        serializable = results.serilizable()
        serializable['func_list'] = list(collect)
        f.write(json.dumps(serializable, indent=2))
if __name__ == '__main__':
    crawl()
