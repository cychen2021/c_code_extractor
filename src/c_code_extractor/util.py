from typing import Sequence

def line_column_to_offset(line: int, column: int, line_sizes: Sequence[int]) -> int:
    return sum(line_sizes[:line]) + column

def offset_to_line_column(offset: int, line_sizes: Sequence[int]) -> tuple[int, int]:
    line = 0
    while line < len(line_sizes) and offset >= line_sizes[line]:
        offset -= line_sizes[line]
        line += 1
    return line, offset
