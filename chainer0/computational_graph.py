import numpy as np
import heapq

dot_edge = '{} -> {} [color=gray30];\n'.format
dot_function_node = '{} [label="{}", shape=box, color=lightblue, style=filled];\n'.format
dot_variable_node = '{} [label="{}", color=orange, style=filled];\n'.format
dot_graph = 'digraph G {{{}}}'.format


def _get_variable_name(v):
    ret = ''
    if v.name is not None:
        ret += v.name + ": "

    shape = v.shape
    if len(shape) == 0:
        ret += 'scalar'
    elif len(shape) == 1:
        ret += '(' + str(shape[0]) + ')'
    else:
        ret += str(shape)
    #ret += ', ' + str(v.data.dtype)
    return ret

def _add_variable_graph(v):
    return dot_variable_node(id(v), _get_variable_name(v))

def _add_function_graph(f):
    ret = dot_function_node(id(f), f.__class__.__name__)
    for input in f.inputs:
        ret += dot_edge(id(input), id(f))
    for output in f.outputs:
        ret += dot_edge(id(f), id(output))
    return ret


def get_dot_graph(output_var):
    cand_funcs = []
    seen_set = set()
    txt = ''
    txt += _add_variable_graph(output_var)

    def add_cand(cand):
        if cand not in seen_set:
            heapq.heappush(cand_funcs, (-cand.rank, len(seen_set), cand))
            seen_set.add(cand)

    add_cand(output_var.creator)

    while cand_funcs:
        _, _, func = heapq.heappop(cand_funcs)
        txt += _add_function_graph(func)
        for x in func.inputs:
            txt += _add_variable_graph(x)

            if x.creator is not None:
                add_cand(x.creator)

    return dot_graph(txt)


