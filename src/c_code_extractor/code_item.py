from typing import Literal

class CodeItem:
    def __init__(self, kind: Literal['typedef', 'funcdef', 'vardef', 'macrodef'], name: str, file: str, start_point: tuple[int, int], end_point: tuple[int, int]):
        self.kind = kind
        self.name = name
        self.file = file
        self.start_point = start_point
        self.end_point = end_point
        
    def __repr__(self) -> str:
        return f'{self.kind} {self.name} in {self.file} [({self.start_point[0]}, {self.start_point[1]}) - ({self.end_point[0]}, {self.end_point[1]})]'

    def __str__(self) -> str:
        return f'{self.kind} {self.name} in {self.file} [({self.start_point[0]}, {self.start_point[1]}) - ({self.end_point[0]}, {self.end_point[1]})]'
        
def read_range(content: str, start_point: tuple[int, int], end_point: tuple[int, int]) -> str:
    start_line, start_col = start_point
    end_line, end_col = end_point
    lines = content.splitlines()
    return '\n'.join(lines[start_line:end_line+1])
