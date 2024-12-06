from lsp_server import LSPServer, path_to_uri
from code_item import CodeItem
import os.path
import json
from typing import Any, Sequence
from ast_analyze import Point

class Edit:
    def __init__(self, new_text: str, start_point: Point, end_point: Point) -> None:
        self.new_text = new_text
        self.start_point = start_point
        self.end_point = end_point
    def __str__(self) -> str:
        return f'{self.new_text=}, {self.start_point=}, {self.end_point=}'
    def __repr__(self) -> str:
        return str(self)

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
            file_path = command['file']
            
            self.lsp_server.notify_open(file_path, 'c')

        self.semantic_tokens_mapping: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    
    def __del__(self) -> None:
        for command in self.compilation_commands:
            self.lsp_server.notify_close(command['file'])
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
    def get_semantic_tokens_in_file(self, file: str) -> dict[tuple[int, int], dict[str, Any]]:
        if file not in self.semantic_tokens_mapping:
            response = self.lsp_server.request_semantic_tokens(file)['result']
            mapping = self.lsp_server.compute_semantic_token_mapping(response['data'])
            self.semantic_tokens_mapping[file] = mapping
        return self.semantic_tokens_mapping[file]

    def get_semantic_tokens_in_range(self, file: str, start_point: Point, end_point: Point) -> dict[tuple[int, int], dict[str, Any]]:
        tokens = self.get_semantic_tokens_in_file(file)
        result = {}
        for key, value in tokens.items():
            if start_point[0] <= key[0] <= end_point[0] and start_point[1] <= key[1] <= end_point[1]:
                result[key] = value
        return result 
        
    def get_semantic_token(self, file: str, start_point: Point) -> dict[str, Any] | None:
        return self.get_semantic_tokens_in_file(file).get((start_point[0], start_point[1]), None)

    def get_function_by_name(self, name: str, file: str):
        response = self.lsp_server.request_all_symbols(file)['result']
        
        result = None
        for symbol in response:
            match self._get_kind_name(symbol['kind']):
                case 'function':
                    if symbol['name'] == name:
                        start_point = symbol['location']['range']['start']['line'], symbol['location']['range']['start']['character']
                        end_point = symbol['location']['range']['end']['line'], symbol['location']['range']['end']['character']
                        result = CodeItem('funcdef', os.path.join(self.lsp_server.cwd, file) , start_point, end_point, name=name)
                case _: pass
            
        return result
    
    # XXX: Clangd's goto definition only gives the header of the definition
    def get_definition(self, file: str, line: int, character: int):
        response = self.lsp_server.request_definition(file, line, character)
        assert 'result' in response, f'{response=}, {file=}, {line=}, {character=}'
        return response['result']

    def get_macro_expansion(self, file: str, start_point: Point, end_point: Point) -> Sequence[Edit]:
        response = self.lsp_server.request_code_action(file, start_point, end_point)
        code_actions = response['result']
        
        expand_macro_action = None
        for code_action in code_actions:
            if code_action['arguments'][0]['tweakID'] == 'ExpandMacro':
                expand_macro_action = code_action
                break
        if expand_macro_action is None:
            return []
        
        expansion = self.lsp_server.request_execute_command(expand_macro_action['command'], expand_macro_action['arguments'])
        if expansion is None:
            return []
        all_changes = expansion['params']['edit']['changes']
        file_uri = path_to_uri(file)
        result = []
        for change in all_changes[file_uri]:
            start_point = Point((change['range']['start']['line'], change['range']['start']['character']))
            end_point = Point((change['range']['end']['line'], change['range']['end']['character']))
            result.append(Edit(change['newText'], start_point, end_point))
        return result

    def refresh_file_content(self, file: str, content: str):
        self.lsp_server.notify_change(file, content)
        self.semantic_tokens_mapping.pop(file)
        with open(file, 'w') as f:
            f.write(content)