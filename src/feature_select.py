import pandas as pd
import numpy as np


def extract_significant_links(p_matrix, val_matrix, dataframe, target_col, alpha=0.05):
    links = []
    N = p_matrix.shape[0]  # 변수 개수

    for i in range(N):  # Effect
        if dataframe.var_names[i] != target_col:
            continue
        for j in range(N):  # Cause
            for tau in range(1, p_matrix.shape[2]):  # 시간차
                p_val = p_matrix[i, j, tau]
                if p_val < alpha:  # 유의미한 경우만 추가
                    links.append((dataframe.var_names[j],  # Cause
                                  dataframe.var_names[i],  # Effect
                                  -tau,  # Lag
                                  val_matrix[i, j, tau],  # Causal Effect
                                  p_val))  # p-value

    return pd.DataFrame(links, columns=["Cause", "Effect", "Lag", "Causal Effect", "p-value"])


class FeatureSelector:
    def __init__(self, data, target_col, method="CFS"):
        self.data = data
        self.target_col = target_col
        self.method = method

    def _select_features_lasso(self, threshold=0.1):
        data = self.data
        features = list()
        return features

    def _select_features_var(self, threshold=0.1):
        results = pd.DataFrame()
        features = list()

        return features

    def _select_features_pcmci(self, threshold=0.1):
        import time
        from tigramite import data_processing as pp  # 전처리
        from tigramite.independence_tests.parcorr import ParCorr  # 조건부 독립성 검정
        from tigramite.pcmci import PCMCI
        start = time.time()
        data = self.data
        target_col = self.target_col

        dataframe = pp.DataFrame(data.values, var_names=list(data.columns))
        ind_test = ParCorr()
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ind_test)

        # pcmci.run_pcmci : PCMCI 실행 후 결과 반환
        # tau_max = 3 : 최대 3개의 과거 시점(t-3까지) 고려
        results = pcmci.run_pcmci(tau_max=10)

        ### 결과 출력
        # pcmci.print_significant_links : 유의미한 인과 관계만 필터링해 출력
        pcmci.print_significant_links(
            p_matrix=results['p_matrix'],
            val_matrix=results['val_matrix'],
            alpha_level=0.05
        )
        end = time.time()
        elpased = end - start
        print(f"{self.method} : {elpased} seconds")
        significant_links_df = extract_significant_links(results['p_matrix'], results['val_matrix'], dataframe,
                                                         target_col)
        return significant_links_df

    def _select_features_pcmciplus(self, threshold=0.1):
        data = self.data
        target_col = self.target_col
        N = data.shape[1]
        from tigramite.jpcmciplus import JPCMCIplus
        from tigramite import data_processing as pp  # 전처리
        from tigramite.independence_tests.parcorr import ParCorr  # 조건부 독립성 검정

        # node_classification을 올바른 딕셔너리 형식으로 변환
        dataframe = pp.DataFrame(data.values, var_names=list(data.columns))
        ind_test = ParCorr()

        node_classification = {i: "system" for i in range(N)}  # 모든 노드를 "system"으로 설정

        # JPCMCIplus 실행
        jpcmci_plus = JPCMCIplus(
            dataframe=dataframe,
            cond_ind_test=ind_test,
            node_classification=node_classification
        )
        # 최대 타임라그 설정
        results_plus = jpcmci_plus.run_pcmciplus(tau_max=3)

        # 결과 출력
        jpcmci_plus.print_significant_links(
            p_matrix=results_plus['p_matrix'],
            val_matrix=results_plus['val_matrix'],
            alpha_level=0.05
        )
        significant_links_df = extract_significant_links(results_plus['p_matrix'], results_plus['val_matrix'],
                                                         dataframe, target_col)
        return significant_links_df

    def _select_features_nbcb(self, threshold=0.1):
        from src.nbcb_e import NBCBe

        nbcb = NBCBe(
            data=self.data,
            tau_max=4,
            sig_level=0.05,
            linear=True,
            model="linear",
            indtest="linear",
            cond_indtest="linear"
        )
        nbcb.run()

        target = self.target_col
        links = []
        if target in nbcb.window_causal_graph_dict:
            for (cause, lag) in nbcb.window_causal_graph_dict[target]:
                links.append({"Cause": cause, "Effect": target, "Lag": lag})
        else:
            print(f"Target {target} not found in NBCB result.")

        df = pd.DataFrame(links, columns=["Cause", "Effect", "Lag"])
        return df


    def _select_features_varlingam(self, threshold=0.01):
        from lingam import VARLiNGAM
        import time

        start = time.time()
        data= self.data

        model = VARLiNGAM()
        model.fit(data.values)

        adjacency_mats = model.adjacency_matrices_ #P_value는 구할 수 없는건가?

        var_names = list(data.columns)
        rows = []

        for lag, mat in enumerate(adjacency_mats, start=1):
            n = mat.shape[0]
            for i in range(n): #Effect (결과 변수)
                for j in range(n): #Cause (원인 변수)
                    effect = mat[i, j]
                    if abs(effect) > threshold:
                        rows.append({
                            "from": f"{var_names[j]}(t-{lag})",
                            "to": f"{var_names[i]}(t)",
                            "effect": effect,
                        })

        # 3) DataFrame으로 반환
        df = pd.DataFrame(rows, columns=["from", "to", "effect"])

        #Com_Gold에 영향을 주는 변수 필터링
        df_com_gold = df[df["to"] == "Com_Gold(t)"]

        end = time.time()
        print(f"{self.method} : {end - start} seconds")

        return df_com_gold


    def select_features(self):
        # TODO Implement Lasso, VAR, NBCB
        # TODO modify PCMCIPlUS
        if self.method == "Lasso":
            return self._select_features_lasso()
        elif self.method == "VAR":
            return self._select_features_var()
        elif self.method == "PCMCIPlUS": #PCMCI, PCMCI+ 구현 완료
            return self._select_features_pcmciplus()
        elif self.method == "VARLiNGAM":
            return self._select_features_varlingam()
        elif self.method == "NBCB":
            return self._select_features_nbcb()
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")
