import pandas as pd
import numpy as np

class FeatureSelector:
    def __init__(self, data, target_col, method="CFS"):
        self.data = data
        self.target_col = target_col
        self.method = method

    def _select_features_lasso(self, threshold=0.1):
        # Placeholder: Lasso 기반 feature selection 구현
        return []

    def _select_features_var(self, threshold=0.1):
        # Placeholder: VAR 모델 기반 feature selection 구현
        return []

    def _select_features_pcmci(self, threshold=0.1):
        import time
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
        start = time.time()
        dataframe = pp.DataFrame(self.data.values, var_names=list(self.data.columns))
        ind_test = ParCorr()
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ind_test)
        results = pcmci.run_pcmci(tau_max=3)
        pcmci.print_significant_links(
            p_matrix=results['p_matrix'],
            val_matrix=results['val_matrix'],
            alpha_level=0.05
        )
        end = time.time()
        print(f"{self.method} : {end - start} seconds")
        # 결과 처리 예시 (원하는 형태로 수정 가능)
        return results

    def _select_features_pcmciplus(self, threshold=0.1):
        from tigramite.jpcmciplus import JPCMCIplus
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        dataframe = pp.DataFrame(self.data.values, var_names=list(self.data.columns))
        ind_test = ParCorr()
        node_classification = {i: "system" for i in range(self.data.shape[1])}
        jpcmci_plus = JPCMCIplus(
            dataframe=dataframe,
            cond_ind_test=ind_test,
            node_classification=node_classification
        )
        results_plus = jpcmci_plus.run_pcmciplus(tau_max=3)
        jpcmci_plus.print_significant_links(
            p_matrix=results_plus['p_matrix'],
            val_matrix=results_plus['val_matrix'],
            alpha_level=0.05
        )
        return results_plus

    def _select_features_varlingam(self, threshold=0.01):
        from lingam import VARLiNGAM
        import time
        start = time.time()
        model = VARLiNGAM()
        model.fit(self.data.values)
        adjacency_mats = model.adjacency_matrices_
        var_names = list(self.data.columns)
        rows = []
        for lag, mat in enumerate(adjacency_mats, start=1):
            n = mat.shape[0]
            for i in range(n):
                for j in range(n):
                    effect = mat[i, j]
                    if abs(effect) > threshold:
                        rows.append({
                            "from": f"{var_names[j]}(t-{lag})",
                            "to": f"{var_names[i]}(t)",
                            "effect": effect,
                        })
        df = pd.DataFrame(rows, columns=["from", "to", "effect"])
        df_com_gold = df[df["to"].str.startswith("Com_Gold")]
        end = time.time()
        print(f"{self.method} : {end - start} seconds")
        return df_com_gold

    def _select_features_nbcb(self, threshold=0.1):
        from src.nbcb import NBCBw

        nbcb = NBCBw(
            data=self.data,
            tau_max=3,
            sig_level=0.05,
            linear=True,
        )
        result = nbcb.run()

        if result is None:
            raise ValueError("Error: NBCBw.run() returned None. Check the function implementation!")

        return result

    def select_features(self):
        if self.method == "Lasso":
            return self._select_features_lasso()
        elif self.method == "VAR":
            return self._select_features_var()
        elif self.method == "PCMCIPlUS":
            return self._select_features_pcmciplus()
        elif self.method == "VARLiNGAM":
            return self._select_features_varlingam()
        elif self.method == "NBCB":
            return self._select_features_nbcb()
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")
