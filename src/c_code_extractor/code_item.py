from typing import Literal

class CodeItem:
    def __init__(self, kind: Literal['init_func', 'typedef', 'funcdef', 'vardef', 'macrodef', 'include_file', 'macro_expand'], file: str,
                 start_point: tuple[int, int] = (-1, -1), end_point: tuple[int, int] = (-1, -1), name: str | None = None):
        self.kind: Literal['init_func', 'typedef', 'funcdef', 'vardef', 'macrodef', 'include_file', 'macro_expand'] = kind
        self.name = name
        self.file = file
        self.start_point = start_point
        self.end_point = end_point
        
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, CodeItem):
            return False
        return self.file == value.file and self.start_point == value.start_point and self.end_point == value.end_point
    
    def __hash__(self) -> int:
        return hash((self.file, self.start_point, self.end_point))
    
    def toJson(self) -> dict:
        match self.kind:
            case 'include_file':
                return {
                    'kind': self.kind,
                    'file': self.file,
                }
            case _:
                return {
                    'kind': self.kind,
                    'name': self.name,
                    'file': self.file,
                    'start_point': self.start_point,
                    'end_point': self.end_point
                } if self.name is not None else {
                    'kind': self.kind,
                    'file': self.file,
                    'start_point': self.start_point,
                    'end_point': self.end_point
                }
        
    def __repr__(self) -> str:
        return f'{self.kind} {self.name + " " if self.name is not None else ""}in {self.file} [({self.start_point[0]}, {self.start_point[1]}) - ({self.end_point[0]}, {self.end_point[1]})]'

    def __str__(self) -> str:
        return f'{self.kind} {self.name + " " if self.name is not None else ""}in {self.file} [({self.start_point[0]}, {self.start_point[1]}) - ({self.end_point[0]}, {self.end_point[1]})]'
        
def read_range(content: str, start_point: tuple[int, int], end_point: tuple[int, int]) -> str:
    start_line, start_col = start_point
    end_line, end_col = end_point
    lines = content.splitlines()
    return '\n'.join(lines[start_line:end_line+1])
