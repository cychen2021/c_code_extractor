from tree_sitter import Parser, Language, Node
import tree_sitter_c as tsc
from code_item import CodeItem

Point = tuple[int, int]
C_LANG = Language(tsc.language())

def point_lt(p1: Point, p2: Point) -> bool:
    if p1[0] < p2[0]:
        return True
    if p1[0] == p2[0] and p1[1] < p2[1]:
        return True
    return False

def point_eq(p1: Point, p2: Point) -> bool:
    return p1[0] == p2[0] and p1[1] == p2[1]

def point_le(p1: Point, p2: Point) -> bool:
    return point_lt(p1, p2) or point_eq(p1, p2)

def is_within(range_inner: tuple[Point, Point], range_outer: tuple[Point, Point]) -> bool:
    inner_start, inner_end = range_inner
    outer_start, outer_end = range_outer
    return point_le(outer_start, inner_start) and point_le(inner_end, outer_end)


def get_ast_of_func(file_path: str, start_point: Point, end_point: Point) -> Node | None:
    parser = Parser(C_LANG)
    with open(file_path, 'r') as f:
        content = f.read()
    ast = parser.parse(content.encode())
    def locate(predicate, args, pattern_index, captures):
        match predicate:
            case 'locate?':
                func_node: Node = captures['func'][0]
                return is_within(
                    (start_point, end_point),
                    ((func_node.start_point.row, func_node.start_point.column), 
                     (func_node.end_point.row, func_node.end_point.column))
                )
            case _:
                return True
        
    query = C_LANG.query(
        r'((function_definition) @func (#locate?))'
    )
    matches = query.matches(ast.root_node, predicate=locate)
    # assert len(matches) == 1
    if len(matches) == 0:
        return None
    nl = matches[0][1]['func']
    assert len(nl) == 1
    return nl[0]

def collect_calls(node: Node) -> list[tuple[str, Point, Point]]:
    query = C_LANG.query(
        r'(call_expression function: (_)@func)'
    )
    matches = query.matches(node)
    result = []
    for m in matches:
        nodes = m[1]['func']
        for n in nodes:
            assert n.text is not None
            result.append((n.text.decode(), 
                           (n.start_point.row, n.start_point.column), 
                           (n.end_point.row, n.end_point.column)))
    return result

def collect_types(node: Node) -> list[tuple[str, Point, Point]]:
    query = C_LANG.query(
        r'(type_identifier)@ty'
    )
    matches = query.matches(node)
    result = []
    for m in matches:
        nodes = m[1]['ty']
        for n in nodes:
            assert n.text is not None
            result.append((n.text.decode(), 
                           (n.start_point.row, n.start_point.column), 
                           (n.end_point.row, n.end_point.column)))
    return result

def collect_identifiers(node: Node) -> list[tuple[str, Point, Point]]:
    query = C_LANG.query(
        r'(identifier)@id'
    )
    matches = query.matches(node)
    result = []
    for m in matches:
        nodes = m[1]['id']
        for n in nodes:
            assert n.text is not None
            result.append((n.text.decode(), 
                           (n.start_point.row, n.start_point.column), 
                           (n.end_point.row, n.end_point.column)))
    return result
