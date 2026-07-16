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

explainer = shapiq.TabularExplainer(
    model=_or,
    data=np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
    ]),
    index="Rred",
    max_order=2,
)

interaction_values = explainer.explain(np.array([1,1,0]), budget=256)

print(interaction_values)
