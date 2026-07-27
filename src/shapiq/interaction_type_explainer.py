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
                if self.sign(value1) == self.sign(value2) == self.sign(values_combined):
                    result.append("synergy")
                elif self.index == "Rred" and valuesrred[interaction_index] > 0:
                    result.append("redundancy")
                elif self.index == "RI" and valuesrred[interaction_index] < 0:
                    result.append("redundancy")
                else:
                    result.append("antagonism")
                interaction_index += 1
        return result
