from tree_sitter import Parser, Language, Node
import tree_sitter_c as tsc
from code_item import CodeItem
from functools import cmp_to_key

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

def point_cmp(p1: Point, p2: Point) -> int:
    if point_lt(p1, p2):
        return -1
    if point_eq(p1, p2):
        return 0
    return 1

def is_within(range_inner: tuple[Point, Point], range_outer: tuple[Point, Point]) -> bool:
    inner_start, inner_end = range_inner
    outer_start, outer_end = range_outer
    return point_le(outer_start, inner_start) and point_le(inner_end, outer_end)

def get_all_funcs(file_path: str) -> list[CodeItem]:
    parser = Parser(C_LANG)
    with open(file_path, 'r') as f:
        content = f.read()
    ast = parser.parse(content.encode())
    query = C_LANG.query(
        r'''(function_definition 
            declarator: (function_declarator
                declarator: (identifier) @func_name
            )
        ) @func'''
    )
    matches = query.matches(ast.root_node)
    result = []
    for m in matches:
        nodes = m[1]['func']
        assert len(nodes) == 1
        n = nodes[0]
        func_names = m[1]['func_name']
        assert len(func_names) == 1
        name = func_names[0].text.decode() # type: ignore
        result.append(CodeItem('funcdef', file_path, 
                                (n.start_point.row, n.start_point.column), 
                                (n.end_point.row, n.end_point.column), name=name))
    return result

def get_ast_of_func_exact_match(file_path: str, start_point: Point, name: str) -> Node | None:
    parser = Parser(C_LANG)
    with open(file_path, 'r') as f:
        content = f.read()
    ast = parser.parse(content.encode())
    def locate(predicate, args, pattern_index, captures):
        match predicate:
            case 'locate?':
                func_node: Node = captures['func'][0]
                name_node: Node = captures['func_name'][0]
                assert name_node.text is not None
                result = func_node.start_point.row == start_point[0] and func_node.start_point.column == start_point[1] \
                    and name_node.text.decode() == name
                return result
            case _:
                return True
        
    query = C_LANG.query(
        r'((function_definition declarator: (_ declarator: (identifier) @func_name) ) @func (#locate?))'
    )
    matches = query.matches(ast.root_node, predicate=locate)
    if len(matches) == 0:
        return None
    nl = matches[0][1]['func']
    assert len(nl) == 1
    return nl[0]

def get_ast_exact_match(file_path: str, start_point: Point, end_point: Point) -> Node | None:
    parser = Parser(C_LANG)
    with open(file_path, 'r') as f:
        content = f.read()
    ast = parser.parse(content.encode())
    def locate(predicate, args, pattern_index, captures):
        match predicate:
            case 'locate?':
                item_node: Node = captures['item'][0]
                result = item_node.start_point.row == start_point[0] and item_node.start_point.column == start_point[1] \
                    and item_node.end_point.row == end_point[0] and item_node.end_point.column == end_point[1]
                return result
            case _:
                return True
        
    query = C_LANG.query(
        r'((_) @item (#locate?))'
    )
    matches = query.matches(ast.root_node, predicate=locate)
    if len(matches) == 0:
        return None
    nl = matches[0][1]['item']
    assert len(nl) == 1
    return nl[0]

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
    if len(matches) == 0:
        return None
    nl = matches[0][1]['func']
    assert len(nl) == 1
    return nl[0]

def get_func_header_from_def(ast: Node) -> str:
    query = C_LANG.query(
        r'(function_definition type: (_) @ty declarator: (_) @decl)'
    )
    matches = query.matches(ast)
    assert len(matches) == 1
    ty = matches[0][1]['ty']
    decl = matches[0][1]['decl']
    assert ty[0].text is not None
    assert decl[0].text is not None
    return f'{ty[0].text.decode()} {decl[0].text.decode()}'

def get_code_item(file_path: str, start_point: Point, end_point: Point) -> CodeItem | None:
    parser = Parser(C_LANG)
    with open(file_path, 'r') as f:
        content = f.read()
    
    ast = parser.parse(content.encode())
    def locate(predicate, args, pattern_index, captures):
        match predicate:
            case 'locate?':
                item_node: Node = captures['item'][0]
                return is_within(
                    (start_point, end_point),
                    ((item_node.start_point.row, item_node.start_point.column), 
                     (item_node.end_point.row, item_node.end_point.column))
                )
            case _:
                return True
    query_func = C_LANG.query(
        r'((function_definition) @item (#locate?))'
    )
    
    query_type = C_LANG.query(
        r'([(struct_specifier) (type_definition)] @item (#locate?))'
    )
    
    query_var = C_LANG.query(
        r'((declaration) @item (#locate?))'
    )
    
    query_macro = C_LANG.query(
        r'([(preproc_def) (preproc_function_def)] @item (#locate?))'
    )

    query_exp = C_LANG.query(
        r'((expression_statement) @item (#locate?))'
    )
    
    func_matches = query_func.matches(ast.root_node, predicate=locate)
    type_matches = query_type.matches(ast.root_node, predicate=locate)
    var_matches = query_var.matches(ast.root_node, predicate=locate)
    macro_matches = query_macro.matches(ast.root_node, predicate=locate)
    exp_matches = query_exp.matches(ast.root_node, predicate=locate)
    
    func_matches.sort(key=lambda x: x[1]['item'][0].start_byte)
    type_matches.sort(key=lambda x: x[1]['item'][0].start_byte)
    var_matches.sort(key=lambda x: x[1]['item'][0].start_byte)
    macro_matches.sort(key=lambda x: x[1]['item'][0].start_byte)
    exp_matches.sort(key=lambda x: x[1]['item'][0].start_byte)
    
    candidates = []
    if len(func_matches) > 0:
        func_match = func_matches[0]
        # name_node = func_match[1]['name'][0]
        # assert name_node.text is not None
        candidates.append(
            CodeItem('funcdef', file_path, func_match[1]['item'][0].start_point, func_match[1]['item'][0].end_point)
                    #  name=name_node.text.decode())
        )
    if len(type_matches) > 0:
        type_match = type_matches[0]
        candidates.append(
            CodeItem('typedef', file_path, type_match[1]['item'][0].start_point, type_match[1]['item'][0].end_point)
        )
    if len(var_matches) > 0:
        var_match = var_matches[0]
        candidates.append(
            CodeItem('vardef', file_path, var_match[1]['item'][0].start_point, var_match[1]['item'][0].end_point)
        )
    if len(macro_matches) > 0:
        macro_match = macro_matches[0]
        candidates.append(
            CodeItem('macrodef', file_path, macro_match[1]['item'][0].start_point, macro_match[1]['item'][0].end_point)
        )
    if len(exp_matches) > 0:
        exp_match = exp_matches[0]
        candidates.append(
            CodeItem('macro_expand', file_path, exp_match[1]['item'][0].start_point, exp_match[1]['item'][0].end_point)
        )    
    
    if len(candidates) == 0:
        return None
    candidates.sort(key=lambda x: cmp_to_key(point_cmp)(x.start_point))
    return candidates[0]
    

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
        r'[(type_identifier) (primitive_type)]@ty'
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
        r'[(identifier) (null)]@id'
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

def collect_declaration_identifiers(node: Node) -> list[tuple[str, Point, Point]]:
    query = C_LANG.query(
        r'''([
            (declaration declarator: (init_declarator declarator: (identifier)@id))
            (declaration declarator: (identifier)@id)
            (pointer_declarator (identifier)@id)
        ])'''
    )
    matches = query.matches(node)
    result = set()
    for m in matches:
        nodes = m[1]['id']
        for n in nodes:
            assert n.text is not None
            result.add((n.text.decode(), 
                           (n.start_point.row, n.start_point.column), 
                           (n.end_point.row, n.end_point.column)))
    return list(result)
