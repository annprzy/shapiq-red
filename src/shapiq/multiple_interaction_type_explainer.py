"""
module explaining the interaction types
"""
import numpy as np
from shapiq import TabularExplainer


class TypeExplainer:
    """
    interaction type explainer class
    """

    def __init__(self, x, model, sample_data, index="RI", max_order=2, budget=256):
        self.x = x
        self.model = model
        self.data = sample_data
        self.index = index
        self.budget = budget
        self.max_order = max_order

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
            max_order=self.max_order,
            normalize=False,
            sample_size=len(self.data),
        )

        valuessii = explainersii.explain(self.x, budget=self.budget)

        explainerrred = TabularExplainer(
            model=self.model,
            data=self.data,
            index=self.index,
            max_order=self.max_order,
            normalize=False,
            sample_size=len(self.data),
        )
        valuesrred = explainerrred.explain(self.x, budget=self.budget)
        result = []
        for key, values_combined in valuessii.dict_values.items():
            lista = list(key)
            if(len(lista) > 1):
                singular_result = {
                    "interaction name": key,
                    "synergy": True,
                    "redundancy": [],
                    "antagonism": [],
                    "independence": [],
                }
                for i in lista:
                    if self.sign(valuessii[i]) != self.sign(valuessii[key]):
                        singular_result["synergy"] = False
                if singular_result["synergy"]:
                    result.append(singular_result)
                    continue
                else:
                    for i in lista:
                        rest = lista.copy()
                        rest.remove(i)
                        rest = tuple(rest)
                        value1 = valuessii[tuple([i])]
                        value2 = valuessii[rest]
                        #print(value1, value2, values_combined)
                        if values_combined > -0.00001 and values_combined < 0.00001:
                            singular_result["independence"].append(i)
                        elif self.index == "Rred" and valuesrred[key] > 0:
                            singular_result["redundancy"].append(i)
                        elif self.index == "RI" and valuesrred[key] < 0:
                            singular_result["redundancy"].append(i)
                        else:
                            singular_result["antagonism"].append(i)
                    result.append(singular_result)
        return result
