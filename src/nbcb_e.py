import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr

from src.pcgce import CITCE
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from lingam.var_lingam import VARLiNGAM
from lingam.resit import RESIT
import matplotlib.pyplot as plt

def run_varlingam(data, tau_max):
    model = VARLiNGAM(lags=tau_max, criterion='bic', prune=False)
    model.fit(data)
    order = model.causal_order_
    order = [data.columns[i] for i in order]
    order.reverse()

    order_matrix = pd.DataFrame(np.zeros([data.shape[1], data.shape[1]]),
                                columns=data.columns, index=data.columns, dtype=int)
    for col_i in order_matrix.index:
        for col_j in order_matrix.columns:
            if col_i != col_j:
                index_i = order.index(col_i)
                index_j = order.index(col_j)
                if index_i > index_j:
                    order_matrix.loc[col_i, col_j] = 2
                    order_matrix.loc[col_j, col_i] = 1
    return order_matrix

def run_resit(data):
    from sklearn.gaussian_process import GaussianProcessRegressor
    reg = GaussianProcessRegressor()
    model = RESIT(regressor=reg)
    model.fit(data)
    order = model.causal_order_
    order = [data.columns[i] for i in order]
    order.reverse()

    order_matrix = pd.DataFrame(np.zeros([data.shape[1], data.shape[1]]),
                                columns=data.columns, index=data.columns, dtype=int)
    for col_i in order_matrix.index:
        for col_j in order_matrix.columns:
            if col_i != col_j:
                index_i = order.index(col_i)
                index_j = order.index(col_j)
                if index_i > index_j:
                    order_matrix.loc[col_i, col_j] = 2
                    order_matrix.loc[col_j, col_i] = 1
    return order_matrix

class NBCBe:
    """
    NBCB_e 모델:
      - noise-based 단계: VARLiNGAM (또는 RESIT)으로 인과 순서를 결정
      - constraint-based 단계: tigramite의 JPCMCIplus(PCMCI+)를 이용해 인과 그래프의 skeleton을 추론
    최종적으로 causal_graph와 window_causal_graph_dict에 결과를 저장합니다.
    """
    def __init__(self, data, tau_max, sig_level, linear=True, model="linear", indtest="linear", cond_indtest="linear"):
        self.data = data
        self.tau_max = tau_max
        self.sig_level = sig_level
        self.linear = linear
        self.model = model
        self.indtest = indtest
        self.cond_indtest = cond_indtest

        self.causal_order = None
        self.graph = []
        self.forbidden_orientation = []
        self.window_causal_graph_dict = dict()
        self.nodes_by_name = {}
        list_nodes = []
        for col in data.columns:
            node = GraphNode(col)
            list_nodes.append(node)
            self.nodes_by_name[col] = node
            self.window_causal_graph_dict[col] = []
        self.causal_graph = GeneralGraph(list_nodes)

    def noise_based(self):
        if self.linear:
            self.causal_order = run_varlingam(self.data, self.tau_max)
        else:
            self.causal_order = run_resit(self.data)
        print("Causal Order:\n", self.causal_order)
        list_columns = list(self.causal_order.columns)
        # 원래 조건: self.causal_order[col_j].loc[col_i]==2 and self.causal_order[col_i].loc[col_j]==1
        # 이를 .loc로 수정하여:
        for col_i in list_columns:
            for col_j in list_columns:
                if (self.causal_order.loc[col_j, col_i] == 2) and (self.causal_order.loc[col_i, col_j] == 1):
                    self.forbidden_orientation.append((list_columns.index(col_j), list_columns.index(col_i)))

    def constraint_based(self, bk=True):
        from tigramite import data_processing as pp
        from tigramite.independence_tests.cmiknn import CMIknn
        from tigramite.jpcmciplus import JPCMCIplus

        dataframe = pp.DataFrame(self.data.values, var_names=list(self.data.columns))
        ind_test = CMIknn(k=5) #K값을 몇으로 설정할까?
        N = self.data.shape[1]
        node_classification = {i: "system" for i in range(N)}

        jpcmci_plus = JPCMCIplus(
            dataframe=dataframe,
            cond_ind_test=ind_test,
            node_classification=node_classification
        )

        results_plus = jpcmci_plus.run_pcmciplus(tau_max=self.tau_max)

        jpcmci_plus.print_significant_links(
            p_matrix=results_plus['p_matrix'],
            val_matrix=results_plus['val_matrix'],
            alpha_level=self.sig_level
        )

        # 1) PCMCI+ 결과 요약하는 summary_matrix 만들기
        summary_matrix = pd.DataFrame(
            np.zeros([N, N]),
            columns=self.data.columns,
            index=self.data.columns
        )
        for i in range(N):
            for j in range(N):
                for tau in range(0, self.tau_max + 1):
                    if results_plus["graph"][i, j, tau] == '-->':
                        summary_matrix.loc[self.data.columns[i], self.data.columns[j]] = 1
                    elif results_plus["graph"][i, j, tau] == '<--':
                        summary_matrix.loc[self.data.columns[j], self.data.columns[i]] = 1

        # 2) forbidden_orientation(인과 순서의 정보) 활용
        list_columns = list(self.data.columns)

        for (idx_j, idx_i) in self.forbidden_orientation:
            cause_name = list_columns[idx_j]
            effect_name = list_columns[idx_i]
            # 만약 summary_matrix에서 cause_name->effect_name (즉 summary_matrix.loc[cause_name, effect_name]==1) 이라면
            # 이 방향을 제거(또는 0으로 만든다)
            if summary_matrix.loc[cause_name, effect_name] == 1:
                summary_matrix.loc[cause_name, effect_name] = 0
                # 필요하다면, 반대방향(effect->cause)을 1로 세팅할 수도 있음 (뒤집고 싶을 때)
                # summary_matrix.loc[effect_name, cause_name] = 1

        # 3) summary matrix를 self.causal_graph에 반영하기
        for col_i in self.data.columns:
            for col_j in self.data.columns:
                # 이미 생성된 노드 객체를 재사용:
                node_i = self.nodes_by_name[col_i]
                node_j = self.nodes_by_name[col_j]
                if (summary_matrix.loc[col_i, col_j] == 1) and (summary_matrix.loc[col_j, col_i] == 1):
                    if (not self.causal_graph.is_parent_of(node_i, node_j)) and \
                            (not self.causal_graph.is_parent_of(node_j, node_i)):
                        self.causal_graph.add_edge(
                            Edge(node_i, node_j, Endpoint.ARROW, Endpoint.ARROW))
                elif summary_matrix.loc[col_i, col_j] == 1:
                    if not self.causal_graph.is_parent_of(node_i, node_j):
                        self.causal_graph.add_edge(
                            Edge(node_i, node_j, Endpoint.TAIL, Endpoint.ARROW))
                elif summary_matrix.loc[col_j, col_i] == 1:
                    if not self.causal_graph.is_parent_of(node_j, node_i):
                        self.causal_graph.add_edge(
                            Edge(node_j, node_i, Endpoint.TAIL, Endpoint.ARROW))

        # 4) 최종 edge를 self.window_causal_graph_dict에 저장
        for edge in self.causal_graph.get_graph_edges():
            cause = edge.get_node1().get_name() if edge.get_endpoint1().name == "TAIL" else edge.get_node2().get_name()
            effect = edge.get_node2().get_name() if edge.get_endpoint2().name == "ARROW" else edge.get_node1().get_name()
            self.window_causal_graph_dict[effect].append((cause, 0))

    def run(self):
        print("######## Running Noise-based step ########")
        self.noise_based()
        print("######## Running Constraint-based step ########")
        self.constraint_based()
        print("NBCB_e finished!")
