"""
module explaining the interaction types
"""
import numpy as np
from shapiq import TabularExplainer


class TypeExplainer:
    """
    interaction type explainer class
    """

    def __init__(self, x, model, sample_data, index="Rred", budget=256):
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
    def explain_coalition_interaction(self, coalition1, coalition2):
        """
        explain interaction between two coalitions
        """
        n = len(self.x)
        explainersii = TabularExplainer(
            model=self.model,
            data=self.data,
            index="k-SII",
            max_order=n,
            normalize=False,
            sample_size=len(self.data),
        ).explain(self.x, budget=self.budget)

        valuesri = TabularExplainer(
            model=self.model,
            data=self.data,
            index=self.index,
            max_order=n,
            normalize=False,
            sample_size=len(self.data),
        ).explain(self.x, budget=self.budget)
        value1 = explainersii.dict_values[coalition1]
        value2 = explainersii.dict_values[coalition2]
        mix = tuple(set(coalition1).union(set(coalition2)))
        values_combined = explainersii.dict_values[mix]
        synergy = True
        for i in mix:
            if self.sign(explainersii.dict_values[tuple([i])]) != self.sign(values_combined):
                synergy = False
        if values_combined > -0.00001 and values_combined < 0.00001:
            return "independence"
        elif synergy and self.sign(value1) == self.sign(value2) == self.sign(values_combined):
            return "synergy"
        elif self.index == "Rred" and valuesri.dict_values[mix] > 0:
            return "redundancy"
        elif self.index == "RI" and valuesri.dict_values[mix] < 0:
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
