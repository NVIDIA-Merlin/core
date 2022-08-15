from merlin.dag import Graph, Node
from merlin.dag.selector import ColumnSelector
from merlin.schema.schema import ColumnSchema, Schema


def test_remove_dependencies():
    # Construct a simple graph with a structure like:
    #   ["y"] ----> ["x", "y"] ---\
    #                              --- > ["o"]
    #   ["z"] --------------------/

    # When removing "y", we should see all of the dependencies
    # and parents removed from the list of leaf nodes.

    dep_node = Node(selector=ColumnSelector(["y"]))
    dep_node.input_schema = Schema([ColumnSchema("y")])
    dep_node.output_schema = Schema([ColumnSchema("y")])

    node_xy = Node(selector=ColumnSelector(["x", "y"]))
    node_xy.input_schema = Schema([ColumnSchema("x"), ColumnSchema("y")])
    node_xy.output_schema = Schema([ColumnSchema("z")])

    plus_node = Node(selector=ColumnSelector(["z", "y"]))
    plus_node.input_schema = Schema([ColumnSchema("y"), ColumnSchema("z")])
    plus_node.output_schema = Schema([ColumnSchema("o")])
    plus_node.add_parent(dep_node)
    plus_node.add_parent(node_xy)
    plus_node.add_dependency(dep_node)

    graph_with_dependency = Graph(plus_node)
    assert len(graph_with_dependency.leaf_nodes) == 2
    graph_with_dependency.remove_inputs(["y"])
    assert len(graph_with_dependency.leaf_nodes) == 1
