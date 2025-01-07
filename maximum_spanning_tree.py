import networkx as nx
from complete_dungeon_graph import CompleteDungeonGraph
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, PULP_CBC_CMD

class MaximumSpanningTree:
    def __init__(self, dungeon_graph):
        self._mst, self._cost = self._calculate_mst(dungeon_graph.graph)

    def _calculate_mst(self, graph):
        edges = list(graph.edges(data=True))
        nodes = list(graph.nodes(data=True))
        num_nodes = len(nodes)

        max_weight = max(data["weight"] for _, _, data in edges)
        aux_weights = {(u, v): max_weight - data["weight"] for u, v, data in edges}

        problem = LpProblem("MinimumSpanningTree", LpMinimize)

        edge_vars = {
            (u, v): LpVariable(f"x_{u}_{v}", cat="Binary")
            for u, v, _ in edges
        }

        problem += lpSum(edge_vars[u, v] * aux_weights[u, v] for u, v in aux_weights)

        problem += lpSum(edge_vars.values()) == num_nodes - 1

        flow_vars = {
            (u, v): LpVariable(f"f_{u}_{v}", lowBound=0, cat="Continuous")
            for u, v, _ in edges
        }
        root_node = nodes[0][0]
        for node in graph.nodes:
            in_flow = lpSum(flow_vars[v, u] for v, u, _ in edges if u == node)
            out_flow = lpSum(flow_vars[u, v] for u, v, _ in edges if u == node)
            if node == root_node:
                problem += out_flow - in_flow == num_nodes - 1
            else:
                problem += out_flow - in_flow == -1

        for u, v, _ in edges:
            problem += flow_vars[u, v] <= edge_vars[u, v] * (num_nodes - 1)

        solver = PULP_CBC_CMD(msg=False)
        problem.solve(solver)

        mst_edges = [
            (u, v, data)
            for u, v, data in edges
            if edge_vars[u, v].value() == 1
        ]

        mst_graph = nx.Graph()
        mst_graph.add_nodes_from(nodes)
        mst_graph.add_edges_from((u, v, {"weight": data["weight"]}) for u, v, data in mst_edges)

        total_cost = sum(data["weight"] for _, _, data in mst_edges)

        return mst_graph, total_cost

    def get_mst(self):
        return self._mst

    def get_cost(self):
        return self._cost

    def debug_print_edges(self):
        print("Edges in the MST:")
        for u, v, data in self._mst.edges(data=True):
            print(f"Edge ({u+1}, {v+1}) with weight {data['weight']}")

    def is_tree(self):
        return nx.is_tree(self._mst)
