
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from skmultilearn.dataset import load_dataset
from skmultilearn.dataset import available_data_sets

from mlca import MLCA

# load dataset ------------------------
# available_data_sets()  # list of dataset
data_name = 'emotions'
data, target, feature_names, label_names = load_dataset(data_name, 'undivided')
data = data.toarray()
target = target.toarray()

x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=0, test_size=0.1)
# ------------------------------------

print('Start...')

# MLCA
mlca = MLCA(plambda=50, v_thres=0.6)
mlca.fit(x_train, y_train)
y_pred = mlca.predict(x_test)

# Evaluations
print('# of Nodes: ', mlca.G_.number_of_nodes())
print('Exact Match: ', metrics.accuracy_score(y_test, y_pred))
print('macro F1-score: ', metrics.f1_score(y_test, y_pred, average='macro'))
print('Label Ranking Average Precision: ', metrics.label_ranking_average_precision_score(y_test, y_pred))
print('Hamming Loss: ', metrics.hamming_loss(y_test, y_pred))
print('Ranking Loss: ', metrics.label_ranking_loss(y_test, y_pred))
print('Coverage Error: ', metrics.coverage_error(y_test, y_pred))
print('Finished')






