"""
module explaining the interaction types
"""
from xml.parsers.expat import model

import numpy as np
from shapiq import TabularExplainer
from shapiq.imputer import MarginalImputer
from shapiq.approximator import PermutationSamplingSII
import itertools


class TypeExplainer:
    """
    interaction type explainer class
    """

    def __init__(self, x, model, sample_data, index="RI", budget=1000):
        self.x = x
        self.model = model
        self.data = sample_data
        self.index = index
        self.budget = budget

    def sign(self, a, margin=0.001):
        """
        check sign of a float
        """
        if a + margin > 0:
            return 1
        return -1
    def calculate_main_effect(self, coalition, explainersii):
        """
        calculate main effect of a coalition
        """
        merged_main_effect = 0.0
        for r in range(1, len(coalition) + 1):
            for subset in itertools.combinations(coalition, r):
                key = tuple(sorted(subset))
                merged_main_effect += explainersii.dict_values[key]
        return merged_main_effect
    def explain_coalition_interaction(self, coalition1, coalition2):
        """
        explain interaction between two coalitions
        """
        def create_grouped_game(base_value_function, groups: list[list[int]], n_features: int):
            listed_indices = {idx for g in groups for idx in g}
            full_groups = [list(g) for g in groups] + [
                [idx] for idx in range(n_features) if idx not in listed_indices
            ]

            def grouped_game(coalitions_groups: np.ndarray) -> np.ndarray:
                is_1d = coalitions_groups.ndim == 1
                if is_1d:
                    coalitions_groups = np.atleast_2d(coalitions_groups)

                n_coalitions = coalitions_groups.shape[0]
                coalitions_features = np.zeros((n_coalitions, n_features), dtype=bool)

                for group_idx, feature_indices in enumerate(full_groups):
                    coalitions_features[:, feature_indices] = coalitions_groups[:, group_idx : group_idx + 1]

                res = base_value_function(coalitions_features)
                if is_1d and isinstance(res, np.ndarray):
                    return res.ravel()
                return res

            return grouped_game, full_groups

        imputer = MarginalImputer(model=self.model, data=self.data, random_state=42)
        imputer.fit(self.x)
        base_value_function = imputer.value_function
        n_features = self.data.shape[1]

        my_groups = [list(coalition1), list(coalition2)]

        grouped_game, full_groups = create_grouped_game(
            base_value_function=base_value_function,
            groups=my_groups,
            n_features=n_features
        )
        approximator = PermutationSamplingSII(
            n=len(full_groups), 
            max_order=max(len(coalition1), len(coalition2)), 
            index="k-SII",
            random_state=42,
        )

        explainersii = approximator.approximate(budget=self.budget, game=grouped_game)


        value1 = explainersii.dict_values[(0,)]
        value2 = explainersii.dict_values[(1,)]
        #print(explainersii.dict_values)
        values_combined = explainersii.dict_values[(0, 1)]

        RI = values_combined / (value1 + value2 + values_combined) if (value1 + value2 + values_combined) != 0 else 0
        
        if self.sign(value1) == self.sign(value2) == self.sign(values_combined):
            return "synergy"
        elif values_combined > -0.00001 and values_combined < 0.00001:
            return "independence"
        elif self.index == "RI" and RI < 0:
            return "redundancy"
        else:
            return "antagonism"
    def explain(self):
        """
        explain interaction type by returing an array of their names
        """
        n = len(self.x)
        explainersii = TabularExplainer(
            model=self.model,
            data=self.data,
            index="k-SII",
            max_order=2,
            normalize=False,
            sample_size=len(self.data),
        )

        valuessii = np.asarray(explainersii.explain(self.x, budget=self.budget))

        explainerrred = TabularExplainer(
            model=self.model,
            data=self.data,
            index=self.index,
            max_order=2,
            normalize=False,
            sample_size=len(self.data),
        )
        valuesrred = np.asarray(explainerrred.explain(self.x, budget=self.budget))
        result = []
        interaction_index = n + 1
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                value1 = valuessii[i]
                value2 = valuessii[j]
                values_combined = valuessii[interaction_index]
                # print(value1, value2, values_combined)
                if values_combined > -0.00001 and values_combined < 0.00001:
                    result.append("independence")
                elif self.sign(value1) == self.sign(value2) == self.sign(values_combined):
                    result.append("synergy")
                elif self.index == "Rred" and valuesrred[interaction_index] > 0:
                    result.append("redundancy")
                elif self.index == "RI" and valuesrred[interaction_index] < 0:
                    result.append("redundancy")
                else:
                    result.append("antagonism")
                interaction_index += 1
        return result
