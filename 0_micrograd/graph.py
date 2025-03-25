"""Helper code to visualize expression graphs."""

import os

from graphviz import Digraph


def _trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.prev:
                edges.add((child, v))
                build(child)

    build(root)

    return nodes, edges


def draw_graph(root, file_basename: str = "graph", file_ext: str = "png"):
    nodes, edges = _trace(root)
    dot = Digraph(graph_attr={"rankdir": "LR"})

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n.op:
            dot.node(name=str(id(n)) + n.op, label=n.op)
            dot.edge(str(id(n)) + n.op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    file_name = file_basename + file_ext
    if os.path.exists(file_name):
        os.remove(file_name)

    dot.render(file_basename, format=file_ext, cleanup=True)
