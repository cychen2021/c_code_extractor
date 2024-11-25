from lsp_server import LSPServer
from code_item import CodeItem
import os.path
import json

class ClangD:
    def __init__(self, src, build) -> None:
        self.lsp_server = LSPServer(
            executable='clangd',
            lsp_args=['--compile-commands-dir', build],
            cwd=src,
        )
        self.lsp_server.start()
        with open(os.path.join(build, 'compile_commands.json'), 'r') as f:
            self.compilation_commands = json.load(f)
        self.src = src
        for command in self.compilation_commands:
            file_path = command['file'].removeprefix(os.path.abspath(src) + '/')
            
            self.lsp_server.notify_open(file_path, 'c')
    
    def __del__(self) -> None:
        for command in self.compilation_commands:
            self.lsp_server.notify_close(command['file'].removeprefix(os.path.abspath(self.src) + '/'))
        self.lsp_server.stop()
    
    @staticmethod
    def _get_kind_name(kind):
        match kind:
            case 5 | 23:
                return 'struct'
            case 8:
                return 'field'
            case 12 | 6:
                return 'function'
            case 13:
                return 'global_variable'
            case 10:
                return 'enum'
            case _:
                assert False, f'Unknown kind: {kind}'


    def get_function_by_name(self, name: str, file: str):
        response = self.lsp_server.request_all_symbols(file)['result']
        
        result = None
        for symbol in response:
            match self._get_kind_name(symbol['kind']):
                case 'function':
                    if symbol['name'] == name:
                        start_point = symbol['location']['range']['start']['line'], symbol['location']['range']['start']['character']
                        end_point = symbol['location']['range']['end']['line'], symbol['location']['range']['end']['character']
                        result = CodeItem('funcdef', name, os.path.join(self.lsp_server.cwd, file) , start_point, end_point)
                case _: pass
            
        return result
    
    # XXX: Clangd's goto definition only gives the header of the definition
    def get_definition(self, file: str, line: int, character: int):
        response = self.lsp_server.request_definition(file, line, character)
        return response['result']
