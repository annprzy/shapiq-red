import shapiq
import numpy as np

n_inputs = 2

def _and(coalitions: np.ndarray):
    outputs = np.zeros(coalitions.shape[0])
    for i,row in enumerate(coalitions):
        if row.all() == True:
            outputs[i]=1
        else:
            outputs[i]=0
    #print(outputs)
    return outputs
def _or(coalitions: np.ndarray):
    outputs = np.zeros(coalitions.shape[0])
    for i,row in enumerate(coalitions):
        if row.any() == True:
            outputs[i]=1
        else:
            outputs[i]=0
    print(outputs, coalitions)
    return outputs
def _and_or(coalitions: np.ndarray):
        """
        outputs the result of the custom gate for each coalition of binary inputs.
        """
        outputs = np.zeros(coalitions.shape[0])
        for i, row in enumerate(coalitions):
            if row[0] and row[1] or row[2]:
                outputs[i] = 1
            else:
                outputs[i] = 0
        return outputs

explainer = shapiq.TabularExplainer(
    model=_and,
    data=np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
    ]),
    index="Rred",
    max_order=2,
    normalize=False,
)

interaction_values = explainer.explain(np.array([0,0]), budget=256)

print(interaction_values)
