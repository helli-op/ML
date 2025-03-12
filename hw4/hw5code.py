import numpy as np
from collections import Counter
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ

    sort_index = np.argsort(feature_vector)
    sort_feature = feature_vector[sort_index]
    sort_target = target_vector[sort_index]

    unique = np.unique(sort_feature)
    thresholds = (unique[:-1] + unique[1:]) / 2

    left_cumsum = np.cumsum(sort_target)[np.unique(sort_feature, return_index = True)[1][1:]- 1] 
    right_cumsum = np.cumsum(sort_target[::-1])[len(sort_target) - np.unique(sort_feature, return_index=True)[1][1:] - 1]
    id = np.unique(sort_feature, return_index=True)[1]
    n = len(sort_target)
    n_left = id[1:]
    n_right = n - n_left

    p1_left = left_cumsum / n_left
    p1_right = right_cumsum / n_right
    p0_left = 1 - p1_left
    p0_right = 1 - p1_right

    H_left = 1 - p1_left**2 - p0_left**2
    H_right = 1 - p1_right**2 - p0_right**2

    ginis = -(n_left / n) * H_left - (n_right / n) * H_right

    best_idx = np.argmax(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best




class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=1, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        sub_X = np.array(sub_X)
        sub_y = np.array(sub_y)
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if (self._max_depth is not None) and (depth >= self._max_depth):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        if len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        if len(sub_y) < self._min_samples_leaf:  
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])

                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if (feature_best is None) or (len(sub_X[split]) < self._min_samples_leaf):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}

        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        x = np.array(x)
        if node["type"] == "terminal":
          return node["class"]
        elif self._feature_types[node['feature_split']] == "real":
          if x[node["feature_split"]] < node["threshold"]:
            return self._predict_node(x, node["left_child"])
          else:
            return self._predict_node(x, node["right_child"])
        else:
          if x[node["feature_split"]] in node["categories_split"]:
            return self._predict_node(x, node["left_child"])
          else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def find_best_split(feature_vector, target_vector):
   
    sort_index = np.argsort(feature_vector)
    sort_feature = feature_vector[sort_index]
    sort_target = target_vector[sort_index]
    unique = np.unique(sort_feature)
    thresholds = (unique[:-1] + unique[1:]) / 2  
    mse_best = None
    threshold_best = None
    mses = []

    for i in range(1, len(unique)):
        threshold = thresholds[i - 1]  
        left_mask = sort_feature < threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue

        y_pred_left = sort_target[left_mask].mean()
        y_pred_right = sort_target[right_mask].mean()

        mse_left = np.mean((sort_target[left_mask] - y_pred_left) ** 2)
        mse_right = np.mean((sort_target[right_mask] - y_pred_right) ** 2)

        mse_total = (left_mask.sum() / len(sort_target)) * mse_left + \
                    (right_mask.sum() / len(sort_target)) * mse_right

        mses.append(mse_total)
        if mse_best is None or mse_total < mse_best:
            mse_best = mse_total
            threshold_best = threshold

    return thresholds, mses, threshold_best, mse_best


class LinearRegressionTree():
    def __init__(self, max_depth=None, min_samples_split=1, min_samples_leaf=1, n_quantiles=10):
        
        self._tree = None
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.quantiles = n_quantiles
    
    def find_best_split(self, X, y):
        best_split = None
        min_mse = float('inf')

        n_samples, n_features = X.shape
        for index in range(n_features):
            unique_values = np.unique(X[:, index])
            if len(unique_values) <= 1:
                continue
            thresholds = np.quantile(unique_values, q=np.linspace(0, 1, self.quantiles + 2)[1: -1])
            for threshold in thresholds:
                left_mask = X[:, index] <= threshold
                right_mask = X[:, index] > threshold

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                X_left, y_left = X[left_mask], y[left_mask]
                X_right, y_right = X[right_mask], y[right_mask]

                left_model = LinearRegression().fit(X_left, y_left)
                y_left_pred = left_model.predict(X_left)
                loss_left = mean_squared_error(y_left, y_left_pred)

                right_model = LinearRegression().fit(X_right, y_right)
                y_right_pred = right_model.predict(X_right)
                loss_right = mean_squared_error(y_right, y_right_pred)

                n_left = len(y_left)
                n_right = len(y_right)
                total_mse = (n_left / n_samples) * loss_left + (n_right / n_samples) * loss_right

                if total_mse < min_mse:
                    min_mse = total_mse
                    best_split = {
                        'feature_index': index,
                        'threshold': threshold,
                        'left_loss': loss_left,
                        'right_loss': loss_right,
                        'left_model': left_model,
                        'right_model': right_model
                    }
        
        return best_split
    def _fit_node(self, sub_X, sub_y, node, current_depth=0):
       
        if (len(sub_y) < self.min_samples_split) or (current_depth >= self.max_depth):
            node["type"] = "terminal"
            model = LinearRegression().fit(sub_X, sub_y)
            node["model"] = model
            return

        best_split = self.find_best_split(sub_X, sub_y)
        if best_split is None:
            node["type"] = "terminal"
            model = LinearRegression().fit(sub_X, sub_y)
            node["model"] = model
            return

        node["type"] = "nonterminal"
        node["feature_split"] = best_split['feature_index']
        node["threshold"] = best_split['threshold']
        node["left_model"] = best_split['left_model']
        node["right_model"] = best_split['right_model']

        left_mask = sub_X[:, best_split['feature_index']] <= best_split['threshold']
        right_mask = sub_X[:, best_split['feature_index']] > best_split['threshold']

        node["left_child"], node["right_child"] = {}, {}

        self._fit_node(sub_X[left_mask], sub_y[left_mask], node["left_child"], current_depth + 1)
        self._fit_node(sub_X[right_mask], sub_y[right_mask], node["right_child"], current_depth + 1)

    def _predict_node(self, x, node):
        while node["type"] == "nonterminal":
            feature = node["feature_split"]
            if x[feature] <= node["threshold"]:
                node = node["left_child"]
            else:
                node = node["right_child"]
        
        if node["type"] == "terminal":
            return node["model"].predict([x])[0]

    def fit(self, X, y):
        self.tree = {}
        self._fit_node(X, y, self.tree)

    def predict(self, X):
        return np.array([self._predict_node(x, self.tree) for x in X])
    
    def set_params(self, **params):
    
        for param, value in params.items():
            if param == 'max_depth':
                self.max_depth = value
            elif param == 'min_samples_split':
                self.min_samples_split = value
            elif param == 'min_samples_leaf':
                self.min_samples_leaf = value
            elif param == 'n_quantiles':
                self.n_quantiles = value
        return self
    def get_params(self, deep=True):
        
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'n_quantiles': self.quantiles
        }

