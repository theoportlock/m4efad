#!/home/theop/venv/bin/python3
# SHAP RESIDUALS SCRIPT
# NEEDS FIXING - Just needs to be added as an option to predict.py and to average across the hypercube

import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from shap import KernelExplainer
from sklearn.ensemble import RandomForestClassifier

class Hypercube:
    def __init__(self, n_vertices):
        self.n_vertices = n_vertices
        self.V = [np.array([])] + self.all_subsets(n_vertices)
        self.V_value = {str(v): 0 for v in self.V}
        self.E = []
        self.E_value = {}
        self.partial_gradient = {vertex: {} for vertex in range(n_vertices)}

    def set_vertex_values(self, vertex_values):
        for v in vertex_values:
            self.V_value[v] = vertex_values[v]
        self._calculate_edges()

    def _calculate_edges(self):
        for i, v in enumerate(self.V):
            for _v in self.V[i + 1:]:
                if self._vertices_form_a_valid_edge(v, _v):
                    self.E.append((v, _v))
                    self.E_value[str((v, _v))] = self.V_value[str(_v)] - self.V_value[str(v)]
        for vertex in range(self.n_vertices):
            self.partial_gradient[vertex] = self.E_value.copy()
            for v, _v in self.E:
                is_relevant_edge = (vertex in v and vertex not in _v) or (vertex in _v and vertex not in v)
                if not is_relevant_edge:
                    self.partial_gradient[vertex][str((v, _v))] = 0

    def _vertices_form_a_valid_edge(self, v, _v):
        differ_in_size_by_1 = (abs(len(v) - len(_v)) == 1)
        the_intersection = np.intersect1d(v, _v)
        intersection_is_nonempty = len(the_intersection) > 0
        if len(v) == 0:
            v_is_the_intersection = (len(_v) == 1)
        elif len(_v) == 0:
            _v_is_the_intersection = (len(v) == 1)
        else:
            v_is_the_intersection = np.array_equal(v, the_intersection)
            _v_is_the_intersection = np.array_equal(_v, the_intersection)
        return differ_in_size_by_1 and intersection_is_nonempty and (v_is_the_intersection or _v_is_the_intersection)

    @staticmethod
    def all_subsets(n_elts):
        res = [np.array(list(itertools.combinations(set(range(n_elts)), i))) for i in range(n_elts)]
        res = {i: res[i] for i in range(n_elts)}
        res[n_elts] = np.array([i for i in range(n_elts)]).reshape(1, -1)
        return [res[i][j] for i in range(1, n_elts + 1) for j in range(res[i].shape[0])]


def get_residual(old_cube, new_cube, vertex):
    res = {}
    for e in old_cube.E_value.keys():
        res[e] = old_cube.partial_gradient[vertex][e] - new_cube.E_value[e]
    return res


def residual_norm(old_cube, vertex_values, vertex):
    new_cube = Hypercube(old_cube.n_vertices)
    new_cube.set_vertex_values({str(_vertex): vertex_values[j] for j, _vertex in enumerate(old_cube.V)})
    return sum([abs(r) for r in get_residual(old_cube, new_cube, vertex).values()])


def shap_residuals(model, features, labels, instance):
    explainer = KernelExplainer(lambda x: model.predict_proba(x), features)
    shap_values = explainer.shap_values(instance)[1]

    coalition_estimated_values = {str(np.array([])): 0}
    coalitions = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    for coalition in coalitions:
        synth = pd.DataFrame(explainer.synth_data)
        for feature in coalition:
            synth = synth[synth[feature] == instance[feature]]
        model_mean = np.mean(labels)
        impact = np.mean(explainer.y[synth.index][:, 1]) - model_mean
        coalition_estimated_values[str(coalition)] = impact

    cube = Hypercube(3)
    cube.set_vertex_values(coalition_estimated_values)

    x0 = np.array([0.5] * 7)
    residuals = []
    for i in range(3):
        f = lambda x: residual_norm(cube, np.append(np.array(0), x), i)
        v = minimize(f, x0)
        residuals.append(v.fun)

    return residuals


if __name__ == "__main__":
    # Generate data
    x1 = np.random.randn(500)
    x2 = np.random.randn(500)
    x3 = np.random.randn(500)
    y = np.int0(np.sqrt(x1**2 + x2**2) < 1)
    df = pd.DataFrame({"Y": y, "X1": x1, "X2": x2, "X3": x3})
    features = df.iloc[:, [1, 2, 3]]
    labels = df.iloc[:, 0]

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=25)
    model.fit(features, labels)

    # Select instance and compute residuals
    instance = features.values[1, :]
    residuals = shap_residuals(model, features, labels, instance)
    print(residuals)
