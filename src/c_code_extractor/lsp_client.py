from lsp_server import LSPServer
from code_item import CodeItem

class ClangD:
    def __init__(self, src, build) -> None:
        self.lsp_server = LSPServer(
            executable='clangd',
            lsp_args=['--compile-commands-dir', build],
            cwd=src,
        )
        self.lsp_server.start()
    
    def __del__(self) -> None:
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
        self.lsp_server.notify_open(file, 'c')
        response = self.lsp_server.request_all_symbols(file)['result']
        
        result = None
        for symbol in response:
            match self._get_kind_name(symbol['kind']):
                case 'function':
                    if symbol['name'] == name:
                        start_point = symbol['location']['range']['start']['line'], symbol['location']['range']['start']['character']
                        end_point = symbol['location']['range']['end']['line'], symbol['location']['range']['end']['character']
                        result = CodeItem('funcdef', name, file, start_point, end_point)
                case _: pass
            
        self.lsp_server.notify_close(file)
        return result
