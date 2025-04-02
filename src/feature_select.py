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


    def _select_features_pcmciplus(self, threshold=0.1):
        from src.pcmci_plus import constraint_based

        selector = constraint_based(
            data=self.data,
            tau_max=3,
            alpha=threshold
        )

        causal_features = selector.select_features_pcmci_plus()

        # 변수명만 추출해서 정리
        feature_names = sorted(set([var for var, lag in causal_features]))

        print(f"PCMCI+ selected features for {self.target_col}: {feature_names}")
        return feature_names

    def _select_features_varlingam(self, threshold=0.01):
        from src.varlingam import noise_based

        selector = noise_based(
            data=self.data,
            tau_max=3,
            threshold=threshold,
            target_var=self.target_col
        )

        selector.fit()

        final_features = selector.select_features(return_only_var_names=True) # True -> 이름만

        print(f"[VarLiNGAM] selected features for {self.target_col}: {final_features}")
        return final_features

    def _select_features_nbcb(self, threshold=0.01):
        from src.nbcb import NBCBw

        nbcb = NBCBw(
            data=self.data,
            tau_max=3,
            sig_level=0.05,
            linear=True,
        )
        full_result, com_gold_causes = nbcb.run()

        if full_result is None:
            raise ValueError("Error: NBCBw.run() returned None. Check the function implementation!")

        return {
            "full_result": full_result,
            "com_gold_causes": com_gold_causes
        }

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
