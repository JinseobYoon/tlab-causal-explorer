
########CBNB_e 한번 구현해보기########

import random
import networkx as nx
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression as lr
from pcgce import CITCE  # 반드시 설치되어 있어야 함
from lingam.var_lingam import VARLiNGAM
from lingam.resit import RESIT

import matplotlib.pyplot as plt
1
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

import os, glob


def clear_args(dir_path):
    files = glob.glob(os.path.join(dir_path, 'args/*'))
    for f in files:
        os.remove(f)


def clear_results(dir_path):
    files = glob.glob(os.path.join(dir_path, 'results/*'))
    for f in files:
        os.remove(f)


def process_data(data, nlags):
    nodes_to_temporal_nodes = dict()
    temporal_nodes = []
    for node in data.columns:
        nodes_to_temporal_nodes[node] = []
        for gamma in range(nlags + 1):
            if gamma == 0:
                temporal_node = f"{node}_t"
            else:
                temporal_node = f"{node}_t_{gamma}"
            nodes_to_temporal_nodes[node].append(temporal_node)
            temporal_nodes.append(temporal_node)

    new_data = pd.DataFrame()
    for gamma in range(nlags + 1):
        shifteddata = data.shift(periods=-nlags + gamma)
        new_columns = [nodes_to_temporal_nodes[node][gamma] for node in data.columns]
        shifteddata.columns = new_columns
        new_data = pd.concat([new_data, shifteddata], axis=1, join="outer")
    new_data.dropna(inplace=True)
    return new_data, nodes_to_temporal_nodes, temporal_nodes


def run_varlingam(data, tau_max):
    model = VARLiNGAM(lags=tau_max, criterion='bic', prune=False)
    model.fit(data)
    order = model.causal_order_
    order = [data.columns[i] for i in order]
    order.reverse()
    order_matrix = pd.DataFrame(0, index=data.columns, columns=data.columns, dtype=int)
    for col_i in order_matrix.index:
        for col_j in order_matrix.columns:
            if col_i != col_j:
                index_i = order.index(col_i)
                index_j = order.index(col_j)
                if index_i > index_j:
                    order_matrix.loc[col_i, col_j] = 1  # col_j is parent of col_i
                else:
                    order_matrix.loc[col_j, col_i] = 1
    return order_matrix


def run_resit(data, tau_max):
    from sklearn.gaussian_process import GaussianProcessRegressor
    reg = GaussianProcessRegressor(normalize_y=True)
    model = RESIT(regressor=reg)
    model.fit(data)
    order = model.causal_order_
    order = [data.columns[i] for i in order]
    order.reverse()
    order_matrix = pd.DataFrame(0, index=data.columns, columns=data.columns, dtype=int)
    for col_i in order_matrix.index:
        for col_j in order_matrix.columns:
            if col_i != col_j:
                index_i = order.index(col_i)
                index_j = order.index(col_j)
                if index_i > index_j:
                    order_matrix.loc[col_i, col_j] = 1
                else:
                    order_matrix.loc[col_j, col_i] = 1
    return order_matrix


class CBNBe:
    def __init__(self, data, tau_max, sig_level, linear=True, model="linear",
                 indtest="linear", cond_indtest="linear"):
        self.data = data
        self.tau_max = tau_max
        self.sig_level = sig_level
        self.linear = linear
        self.model = model
        self.indtest = indtest
        self.cond_indtest = cond_indtest

        self.causal_order = []
        self.graph = []  # 최종 causal graph (causallearn의 graph 객체)
        self.forbidden_orientation = []

        # 각 변수별 lagged parent 정보를 저장할 dict
        self.window_causal_graph_dict = {col: [] for col in data.columns}
        self.window_causal_graph = None

        # causallearn의 그래프 초기화
        list_nodes = [GraphNode(col) for col in data.columns]
        self.causal_graph = GeneralGraph(list_nodes)

    def constraint_based(self):
        # CITCE를 사용하여 조건부 독립성 기반 skeleton 추정
        pcgce = CITCE(self.data, sig_lev=self.sig_level, lag_max=self.tau_max, linear=self.linear)
        pcgce.skeleton_initialize()
        pcgce.find_sep_set()

        print("CB 단계: CITCE에서 추정한 skeleton edges:")
        print(pcgce.graph.ghat.edges)

        col_names = list(self.data.columns)
        # 초기 causal graph를 "---"로 채운 3D array (lag 0 ~ tau_max)
        self.window_causal_graph = np.full([len(col_names), len(col_names), self.tau_max + 1], "---", dtype=object)
        map_names_nodes_inv = {}
        for node in pcgce.graph.map_names_nodes:
            for node_t in pcgce.graph.map_names_nodes[node]:
                map_names_nodes_inv[node_t] = node
        # edge를 순회하며 lag 정보 설정 (lag=0는 동시, lag>=1은 시차 효과)
        for edge in pcgce.graph.ghat.edges:
            node_0 = map_names_nodes_inv[edge[0]]
            node_1 = map_names_nodes_inv[edge[1]]
            i = col_names.index(node_0)
            j = col_names.index(node_1)
            # 예시에서는 edge[0]가 현재 시점(t=0)인 경우, lag 1부터 effect로 간주
            # 필요시 아래 조건을 데이터에 맞게 수정
            for t in range(1, self.tau_max + 1):
                self.window_causal_graph[i, j, t] = "-->"
        print("CB 단계 후 window_causal_graph:")
        print(self.window_causal_graph)

    def find_cycle_groups(self):
        instantaneous_nodes = []
        instantaneous_graph = nx.Graph()
        for i in range(len(self.data.columns)):
            for j in range(len(self.data.columns)):
                t = 0  # t=0: 동시 효과
                if self.window_causal_graph[i, j, t] in {"o-o", "x-x", "-->", "<--"}:
                    instantaneous_graph.add_edge(self.data.columns[i], self.data.columns[j])
                    if self.data.columns[i] not in instantaneous_nodes:
                        instantaneous_nodes.append(self.data.columns[i])
        list_cycles = nx.cycle_basis(instantaneous_graph)
        cycle_groups = {}
        idx = 0
        for cycle in list_cycles:
            cycle_groups[idx] = cycle
            idx += 1
        return cycle_groups, list_cycles, instantaneous_nodes

    def noise_based(self):
        # 여기서는 cycle 그룹 내에서 VARLiNGAM (또는 RESIT)로 재정렬해 인과 방향을 재확인
        cycle_groups, list_cycles, instantaneous_nodes = self.find_cycle_groups()
        print("즉시 효과를 가진 노드들:", instantaneous_nodes)
        print("cycle groups:", cycle_groups)

        list_columns = list(self.data.columns)
        if len(instantaneous_nodes) > 1:
            for idx in cycle_groups.keys():
                cyc_nodes = cycle_groups[idx]
                # 나머지 부모 후보: cycle 외의 변수들
                parents_nodes = list(set(list_columns) - set(cyc_nodes))
                sub_data = self.data[cyc_nodes + parents_nodes]
                # 선형 모형 사용 여부에 따라 VARLiNGAM 또는 RESIT 실행
                if self.linear:
                    causal_order = run_varlingam(sub_data, self.tau_max)
                else:
                    causal_order = run_resit(sub_data, self.tau_max)
                print("노이즈 기반 causal order (cycle 그룹):")
                print(causal_order)
                # causal_order에 따라 lag=0에서 방향을 정해준다.
                for col_i in cyc_nodes:
                    for col_j in cyc_nodes:
                        if col_i == col_j:
                            continue
                        if (causal_order.loc[col_i, col_j] == 0) and (causal_order.loc[col_j, col_i] == 1):
                            i = list_columns.index(col_i)
                            j = list_columns.index(col_j)
                            t = 0
                            if self.window_causal_graph[i, j, t] in {"o-o", "x-x", "-->", "<--"}:
                                self.window_causal_graph[i, j, t] = "-->"
                                self.window_causal_graph[j, i, t] = "<--"
                                print(f"방향 수정: {col_i} -> {col_j} at t={t}")

    def construct_summary_causal_graph(self):
        summary_matrix = pd.DataFrame(0, index=self.data.columns, columns=self.data.columns)
        for i in range(len(self.data.columns)):
            for j in range(len(self.data.columns)):
                for t in range(self.tau_max + 1):
                    if self.window_causal_graph[i, j, t] == '-->':
                        if (self.data.columns[i], -t) not in self.window_causal_graph_dict[self.data.columns[j]]:
                            self.window_causal_graph_dict[self.data.columns[j]].append((self.data.columns[i], -t))
                            summary_matrix.loc[self.data.columns[i], self.data.columns[j]] = 1
                    elif self.window_causal_graph[i, j, t] == '<--':
                        if (self.data.columns[j], -t) not in self.window_causal_graph_dict[self.data.columns[i]]:
                            self.window_causal_graph_dict[self.data.columns[i]].append((self.data.columns[j], -t))
                            summary_matrix.loc[self.data.columns[j], self.data.columns[i]] = 1

        # causallearn의 graph 객체에 edge 추가
        for col_i in self.data.columns:
            for col_j in self.data.columns:
                if summary_matrix.loc[col_i, col_j] == 1 and summary_matrix.loc[col_j, col_i] == 1:
                    if (not self.causal_graph.is_parent_of(GraphNode(col_i), GraphNode(col_j)) and
                            not self.causal_graph.is_parent_of(GraphNode(col_j), GraphNode(col_i))):
                        self.causal_graph.add_edge(
                            Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.ARROW, Endpoint.ARROW))
                elif summary_matrix.loc[col_i, col_j] == 1:
                    if not self.causal_graph.is_parent_of(GraphNode(col_i), GraphNode(col_j)):
                        self.causal_graph.add_edge(
                            Edge(GraphNode(col_i), GraphNode(col_j), Endpoint.TAIL, Endpoint.ARROW))
                elif summary_matrix.loc[col_j, col_i] == 1:
                    if not self.causal_graph.is_parent_of(GraphNode(col_j), GraphNode(col_i)):
                        self.causal_graph.add_edge(
                            Edge(GraphNode(col_j), GraphNode(col_i), Endpoint.TAIL, Endpoint.ARROW))
        print("Summary causal graph (edge list):")
        print(self.causal_graph.get_graph_edges())

    def run(self):
        print("######## Running Constraint-based ########")
        self.constraint_based()
        print("######## Running Noise-based ########")
        self.noise_based()
        print("######## Construct summary causal graph ########")
        self.construct_summary_causal_graph()


if __name__ == '__main__':
    # 데이터 불러오기 (예시: csv 파일 등)
    # data = pd.read_csv("your_data.csv")
    # 여기서는 임의의 예시 데이터로 진행합니다.
    np.random.seed(0)
    data = pd.DataFrame(np.random.randn(200, 10), columns=[f"Var{i}" for i in range(10)])

    # 예시: tau_max=3, sig_level=0.05
    cbnb_e = CBNBe(data, tau_max=3, sig_level=0.05, linear=True)
    cbnb_e.run()
