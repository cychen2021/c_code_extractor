import subprocess
import urllib.parse
import os.path
import json
from typing import Any, Literal

def path_to_uri(path):
    return "file://" + os.path.abspath(path)


def uri_to_path(uri):
    data = urllib.parse.urlparse(uri)

    assert data.scheme == "file"
    assert not data.netloc
    assert not data.params
    assert not data.query
    assert not data.fragment

    path = data.path
    if path.startswith(os.getcwd()):
        path = os.path.relpath(path, os.getcwd())
    return urllib.parse.unquote(path)  # clangd seems to escape paths.


def check(executable):
    try:
        # TODO: enforce version
        clangd = subprocess.run(
            [executable, '--version'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return clangd.returncode == 0
    except FileNotFoundError:
        return False
    
class LSPServer:
    class InitResult:
        def __init__(self, semantic_tokens_legend):
            self.semantic_tokens_legend = semantic_tokens_legend
    
    @staticmethod
    def _to_lsp_request(id, method, params):
        request = {"jsonrpc": "2.0", "id": id, "method": method}
        if method == "initialize":
            params["capabilities"] = {}
        if params:
            request["params"] = params

        content = json.dumps(request)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        return header + content


    @staticmethod
    def _to_lsp_notification(method, params):
        request = {"jsonrpc": "2.0", "method": method}
        if params:
            request["params"] = params

        content = json.dumps(request)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        return header + content

    @staticmethod
    def _parse_lsp_response(id, file):
        while True:
            header = {}
            while True:
                line = file.readline().strip()
                if not line:
                    break
                key, value = line.split(":", 1)
                header[key.strip()] = value.strip()

            content = file.read(int(header["Content-Length"]))
            response = json.loads(content)
            if "id" in response and response["id"] == id:
                return response
    
    def __init__(
        self,
        executable,
        lsp_args,
        cwd=os.getcwd(),
    ):
        self.lsp_args = lsp_args
        self.executable = executable
        self.cwd = cwd
        self._process = None
        self._sequence = 0
        self.semantic_tokens_legend = {
            "tokenTypes": [],
            "tokenModifiers": [],
        }
        self.file_version = {}
        self.original_content = {}
        
    def __enter__(self):
        self.start()
        return self

    def restart(self, cwd: str | None = None):
        self.stop()
        self.cwd = cwd or self.cwd
        self.start()
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        
    def __del__(self):
        self.stop()
        
    def initialize_lsp_server(self) -> 'LSPServer.InitResult':
        r = self.request("initialize", {
            'processId': os.getpid(),
            'rootUri': path_to_uri(self.cwd),
        })
        capabilities = r['result']['capabilities']
        semantic_tokens_legend = capabilities.get('semanticTokensProvider', {}).get('legend')
        self.notify("initialized", {})
        return self.InitResult(semantic_tokens_legend)
    
    def get_and_inc_sequence(self):
        seq = self._sequence
        self._sequence += 1
        return seq
    
    def request_hover(self, file_name, line, character):
        return self.request('textDocument/hover', {
            'textDocument': {'uri': path_to_uri(file_name)},
            'position': {
                'line': line,
                'character': character,
            }
        })
    
    def request_code_action(self, file_name, start_point, end_point):
        return self.request('textDocument/codeAction', {
            'textDocument': {'uri': path_to_uri(file_name)},
            'range': {
                'start': {
                    'line': start_point[0],
                    'character': start_point[1],
                },
                'end': {
                    'line': end_point[0],
                    'character': end_point[1],
                }
            },
            'context': {
                'diagnostics': [],
            }
        })
    
    def request_execute_command(self, command, arguments):
        assert command == 'clangd.applyTweak', f'{command=}'
        assert arguments[0]['tweakID'] == 'ExpandMacro', f'{arguments=}'
        return self.request_special(
            'expandMacro',
            'workspace/executeCommand',
            {
                'command': command,
                'arguments': arguments,
            },
            command=command,
        )

    def send_response(self, id, result):
        assert self._process is not None
        assert self._process.stdin is not None
        response = {
            'jsonrpc': '2.0',
            'id': id,
            'result': result,
        }
        content = json.dumps(response)
        header = f'Content-Length: {len(content)}\r\n\r\n'
        self._process.stdin.write(header + content)
        self._process.stdin.flush()

    def request_special(self, kind: Literal['expandMacro'], method, params, **kwargs):
        @staticmethod
        def _wait_apply_edit(file):
            while True:
                header = {}
                while True:
                    line = file.readline().strip()
                    if not line:
                        break
                    key, value = line.split(":", 1)
                    header[key.strip()] = value.strip()

                content = file.read(int(header["Content-Length"]))
                response = json.loads(content)
                if response['method'] == 'workspace/applyEdit':
                    return response
        match kind:
            case 'expandMacro':
                if kwargs['command'] == 'clangd.applyTweak' and params['arguments'][0]['tweakID'] == 'ExpandMacro':
                    assert self._process is not None
                    assert self._process.stdin is not None
                    assert self._process.stdout is not None
                    message_seq = self.get_and_inc_sequence()
                    self._process.stdin.write(self._to_lsp_request(message_seq, method, params))
                    self._process.stdin.flush()
                    edit = _wait_apply_edit(self._process.stdout)
                    edit_id = edit['id']
                    self.send_response(edit_id, {'applied': True})
                    return edit

    def start(self):
        import sys
        self._sequence = 0
        self._process = subprocess.Popen(
            [self.executable, *self.lsp_args],
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=self.cwd,
        )
        r = self.initialize_lsp_server()
        self.semantic_tokens_legend = r.semantic_tokens_legend
    
    def semantic_token_type_of(self, token_type_id: int):
        return self.semantic_tokens_legend['tokenTypes'][token_type_id]
    
    def semantic_token_modifier_of(self, token_modifier_flags: int):
        result = []
        for i in range(token_modifier_flags.bit_length()):
            mask = 1 << i
            flat_set = token_modifier_flags & mask != 0
            if flat_set:
                result.append(self.semantic_tokens_legend['tokenModifiers'][i])
        return result

    def stop(self):
        if self._process is None:
            return
        self._process.terminate()
        
    def request(self, method, params):
        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        message_seq = self.get_and_inc_sequence()
        self._process.stdin.write(self._to_lsp_request(message_seq, method, params))
        self._process.stdin.flush()
        return self._parse_lsp_response(message_seq, self._process.stdout)
    
    def notify(self, method, params):
        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write(self._to_lsp_notification(method, params))
        self._process.stdin.flush()
        
    
    def notify_open(self, filename, language_id):
        with open(os.path.join(self.cwd, filename), "r") as file:
            text = file.read()
        self.file_version[filename] = 1
        self.original_content[filename] = text

        self.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": path_to_uri(filename),
                "languageId": language_id,
                "version": 1,
                "text": text,
            }
        })
    
    def notify_close(self, file_name):
        self.notify("textDocument/didClose", {"textDocument": {"uri": path_to_uri(file_name)}})
    
    def modified_files(self) -> dict[str, str]:
        result = {}
        for file_name, version in self.file_version.items():
            if version > 1:
                result[file_name] = self.original_content[file_name]
        return result
        
    def compute_semantic_token_mapping(self, semantic_token_encoding) -> dict[tuple[int, int], dict[str, Any]]:
        assert len(semantic_token_encoding) % 5 == 0, f'{len(semantic_token_encoding)=}'
        previous_line = 0
        previous_start = 0
        tokens = {}
        for i in range(0, len(semantic_token_encoding), 5):
            delta_line = semantic_token_encoding[i]
            delta_start = semantic_token_encoding[i+1]
            length = semantic_token_encoding[i+2]
            token_type = semantic_token_encoding[i+3]
            token_modifiers = semantic_token_encoding[i+4]
            
            if delta_line != 0:
                previous_start = 0
            
            line = previous_line + delta_line
            start = previous_start + delta_start
            item = {
                'line': line,
                'start': start,
                'length': length,
                'type': self.semantic_token_type_of(token_type),
                'modifiers': self.semantic_token_modifier_of(token_modifiers),
            }
            tokens[(line, start)] = item
            previous_line = line
            previous_start = start
        return tokens

    def request_definition(self, file_name, line, character):
        return self.request("textDocument/definition", {
            "textDocument": {"uri": path_to_uri(file_name)},
            "position": {
                "line": line,
                "character": character,
            }
        })

    def request_type_definition(self, file_name, line, character):
        return self.request("textDocument/typeDefinition", {
            "textDocument": {"uri": path_to_uri(file_name)},
            "position": {
                "line": line,
                "character": character,
            }
        })

    def request_semantic_tokens_in_range(self, file_name, range, start_line, start_character, end_line, end_character):
        return self.request("textDocument/semanticTokens/range", {
            "textDocument": {"uri": path_to_uri(file_name)},
            "range": {
                "start": {
                    "line": start_line,
                    "character": start_character,
                },
                "end": {
                    "line": end_line,
                    "character": end_character,
                }
            }
        })
    
    def request_semantic_tokens(self, file_name):
        return self.request("textDocument/semanticTokens/full", {
            "textDocument": {"uri": path_to_uri(file_name)}
        })
        
    def request_declaration(self, file_name, line, character):
        return self.request("textDocument/declaration", {
            "textDocument": {"uri": path_to_uri(file_name)},
            "position": {
                "line": line,
                "character": character,
            }
        })

    def request_references(self, file_name, line, character):
        return self.request("textDocument/references", {
            "textDocument": {"uri": path_to_uri(file_name)},
            "position": {
                "line": line,
                "character": character,
            },
            "context": {
                "includeDeclaration": True
            }
        })

    def request_signature(self, file_name, line, character):
        return self.request("textDocument/typeDefinition", {
            "textDocument": {"uri": path_to_uri(file_name)},
            "position": {
                "line": line,
                "character": character,
            }
        })

    def request_all_symbols(self, file_name):
        return self.request("textDocument/documentSymbol", {
            "textDocument": {"uri": path_to_uri(file_name)}
        })

    def notify_change(self, file_name, newContent):
        the_file = os.path.abspath(file_name)
        self.file_version[the_file] += 1
        self.notify("textDocument/didChange", {
            "textDocument": {
                "uri": path_to_uri(the_file),
                "version": self.file_version[the_file],
            },
            "contentChanges": [{
                "text": newContent
            }]
        })