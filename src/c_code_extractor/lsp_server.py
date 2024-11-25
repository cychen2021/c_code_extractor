import subprocess
import urllib.parse
import os.path
import json

def _path_to_uri(path):
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
            [executable, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return clangd.returncode == 0
    except FileNotFoundError:
        return False
    
class LSPServer:
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
        
    def initialize_lsp_server(self):
        r = self.request("initialize", {
            'processId': os.getpid(),
            'rootUri': _path_to_uri(self.cwd),
        })
        self.notify("initialized", {})
    
    def get_and_inc_sequence(self):
        seq = self._sequence
        self._sequence += 1
        return seq
    
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
        self.initialize_lsp_server()

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

        self.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": _path_to_uri(filename),
                "languageId": language_id,
                "version": 1,
                "text": text,
            }
        })
    
    def notify_close(self, file_name):
        self.notify("textDocument/didClose", {"textDocument": {"uri": _path_to_uri(file_name)}})

    def request_definition(self, file_name, line, character):
        return self.request("textDocument/definition", {
            "textDocument": {"uri": _path_to_uri(file_name)},
            "position": {
                "line": line,
                "character": character,
            }
        })

    def request_type_definition(self, file_name, line, character):
        return self.request("textDocument/typeDefinition", {
            "textDocument": {"uri": _path_to_uri(file_name)},
            "position": {
                "line": line,
                "character": character,
            }
        })
    def request_declaration(self, file_name, line, character):
        return self.request("textDocument/declaration", {
            "textDocument": {"uri": _path_to_uri(file_name)},
            "position": {
                "line": line,
                "character": character,
            }
        })

    def request_references(self, file_name, line, character):
        return self.request("textDocument/references", {
            "textDocument": {"uri": _path_to_uri(file_name)},
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
            "textDocument": {"uri": _path_to_uri(file_name)},
            "position": {
                "line": line,
                "character": character,
            }
        })

    def request_all_symbols(self, file_name):
        return self.request("textDocument/documentSymbol", {
            "textDocument": {"uri": _path_to_uri(file_name)}
        })