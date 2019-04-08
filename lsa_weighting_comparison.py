"""
    Basic experimentation of document classification with
    1. TF-IDF vs Count Vectors and
    2. Sparse vs. LSA (Truncated SVD) vs. Chisquare Feature Selection settings.

    Dataset - 20 newsgroups corpus
    Classifier - Linear SVC
"""

from joblib import Memory
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from shutil import rmtree
from tempfile import mkdtemp
import time
import warnings

__author__ = 'avijitv'


# Hyper-parameters
MAX_DF = 0.5
MIN_DF = 2
MAX_FEATURES = 1000

BOW_PARAM_GRID = {'vect__max_features': [None],
                  'svc__C': [0.01, 0.1, 1, 10, 100]}
FSELECT_PARAM_GRID = {'vect__max_features': [None],
                      'feature_selection__k': [100, 200, 500, 1000, 1500, 2000],
                      'svc__C': [0.01, 0.1, 1]}
LSA_PARAM_GRID = {'vect__max_features': [None],
                  'svd__n_components': [100, 500],
                  'svc__C': [0.01, 0.1]}


def tune_model(model, param_grid, x, y):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, refit=True,
                               cv=3,
                               scoring='accuracy', verbose=1)
    grid_search.fit(x, y)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print("Total time to refit: %0.3f" % grid_search.refit_time_)

    return grid_search.best_estimator_, grid_search.refit_time_


def main():
    warnings.filterwarnings("ignore")
    print('Loading 20 Newsgroup data')
    data_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42,
                                    remove=('headers', 'footers', 'quotes'))

    data_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42,
                                   remove=('headers', 'footers', 'quotes'))

    train_x, train_y = data_train.data, data_train.target
    test_x, test_y = data_test.data, data_test.target

    print('Number of samples for train :', len(train_x))
    print('Number of samples for test :', len(test_x))
    print()

    performance_stats_lists = {'accuracy': [],
                               'training_time': [],
                               'test_time': []}

    print('BoW Models')
    print('---------------')
    print()
    print('-----')
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=0)
    tf_idf_pipeline = Pipeline([('vect', TfidfVectorizer(max_df=MAX_DF, min_df=MIN_DF, stop_words='english')),
                                ('svc', LinearSVC(dual=True, max_iter=3000))],
                               memory=memory)
    print('Tuning BoW TF-IDF Vectors Pipeline')
    tf_idf_pipeline, tf_idf_train_time = tune_model(tf_idf_pipeline, BOW_PARAM_GRID, train_x, train_y)
    test_time_start = time.time()
    pred_y = tf_idf_pipeline.predict(test_x)
    test_time_end = time.time()
    tf_idf_test_time = test_time_end - test_time_start
    tf_idf_acc = accuracy_score(pred_y, test_y)
    print('Test Accuracy : %0.3f' % tf_idf_acc)
    performance_stats_lists['accuracy'].append(tf_idf_acc)
    performance_stats_lists['training_time'].append(tf_idf_train_time)
    performance_stats_lists['test_time'].append(tf_idf_test_time)
    rmtree(cachedir)
    print('-----')
    print()

    print('-----')
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=0)
    count_pipeline = Pipeline([('vect', CountVectorizer(max_df=MAX_DF, min_df=MIN_DF, stop_words='english')),
                               ('svc', LinearSVC(dual=True, max_iter=3000))],
                              memory=memory)
    print('Tuning BoW Count Vectors Pipeline')
    count_pipeline, count_train_time = tune_model(count_pipeline, BOW_PARAM_GRID, train_x, train_y)
    test_time_start = time.time()
    pred_y = count_pipeline.predict(test_x)
    test_time_end = time.time()
    count_test_time = test_time_end - test_time_start
    count_acc = accuracy_score(pred_y, test_y)
    print('Test Accuracy : %0.3f' % count_acc)
    performance_stats_lists['accuracy'].append(count_acc)
    performance_stats_lists['training_time'].append(count_train_time)
    performance_stats_lists['test_time'].append(count_test_time)
    rmtree(cachedir)
    print('-----')
    print()
    print('---------------')

    print()
    print()

    print('BoW +  Chi-Square Feature Selection Models')
    print('---------------')
    print()
    print('-----')
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=0)
    chi_tf_idf_pipeline = Pipeline([('vect', TfidfVectorizer(max_df=MAX_DF, min_df=MIN_DF, stop_words='english')),
                                    ('feature_selection', SelectKBest(chi2)),
                                    ('svc', LinearSVC(dual=False))],
                                   memory=memory)
    print('Tuning BoW TF-IDF Vectors with Chi-Square feature selection Pipeline')
    chi_tf_idf_pipeline, chi_tf_idf_train_time = tune_model(chi_tf_idf_pipeline, FSELECT_PARAM_GRID, train_x, train_y)
    test_time_start = time.time()
    pred_y = chi_tf_idf_pipeline.predict(test_x)
    test_time_end = time.time()
    chi_tf_idf_test_time = test_time_end - test_time_start
    chi_tf_idf_acc = accuracy_score(pred_y, test_y)
    print('Test Accuracy : %0.3f' % chi_tf_idf_acc)
    rmtree(cachedir)
    performance_stats_lists['accuracy'].append(chi_tf_idf_acc)
    performance_stats_lists['training_time'].append(chi_tf_idf_train_time)
    performance_stats_lists['test_time'].append(chi_tf_idf_test_time)
    print('-----')
    print()

    print('-----')
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=0)
    chi_count_pipeline = Pipeline([('vect', CountVectorizer(max_df=MAX_DF, min_df=MIN_DF, stop_words='english')),
                                   ('feature_selection', SelectKBest(chi2)),
                                   ('svc', LinearSVC(dual=False))],
                                  memory=memory)
    print('Tuning BoW Count Vectors with Chi-square feature selection Pipeline')
    chi_count_pipeline, chi_count_train_time = tune_model(chi_count_pipeline, FSELECT_PARAM_GRID, train_x, train_y)
    test_time_start = time.time()
    pred_y = chi_count_pipeline.predict(test_x)
    test_time_end = time.time()
    chi_count_test_time = test_time_end - test_time_start
    chi_count_acc = accuracy_score(pred_y, test_y)
    print('Test Accuracy : %0.3f' % chi_count_acc)
    rmtree(cachedir)
    performance_stats_lists['accuracy'].append(chi_count_acc)
    performance_stats_lists['training_time'].append(chi_count_train_time)
    performance_stats_lists['test_time'].append(chi_count_test_time)
    print('-----')
    print()

    print('---------------')
    print()
    print()

    print('BoW + LSA Models')
    print('---------------')
    print()
    print('-----')
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=0)
    lsa_tf_idf_pipeline = Pipeline([('vect', TfidfVectorizer(max_df=MAX_DF, min_df=MIN_DF, stop_words='english')),
                                    ('svd', TruncatedSVD(random_state=42)),
                                    ('svc', LinearSVC(dual=False))],
                                   memory=memory)
    print('Tuning LSA with TF-IDF Vectors Pipeline')
    lsa_tf_idf_pipeline, lsa_tf_idf_train_time = tune_model(lsa_tf_idf_pipeline, LSA_PARAM_GRID, train_x, train_y)
    test_time_start = time.time()
    pred_y = lsa_tf_idf_pipeline.predict(test_x)
    test_time_end = time.time()
    lsa_tf_idf_test_time = test_time_end - test_time_start
    lsa_tf_idf_acc = accuracy_score(pred_y, test_y)
    print('Test Accuracy : %0.3f' % lsa_tf_idf_acc)
    rmtree(cachedir)
    performance_stats_lists['accuracy'].append(lsa_tf_idf_acc)
    performance_stats_lists['training_time'].append(lsa_tf_idf_train_time)
    performance_stats_lists['test_time'].append(lsa_tf_idf_test_time)
    print('-----')
    print()

    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=0)
    lsa_count_pipeline = Pipeline([('vect', CountVectorizer(max_df=MAX_DF, min_df=MIN_DF,
                                                            stop_words='english')),
                                   ('svd', TruncatedSVD(random_state=42)),
                                   ('svc', LinearSVC(dual=False))],
                                  memory=memory)
    print('Tuning LSA with Count Vectors Pipeline')
    lsa_count_pipeline, lsa_count_train_time = tune_model(lsa_count_pipeline, LSA_PARAM_GRID, train_x, train_y)
    test_time_start = time.time()
    pred_y = lsa_count_pipeline.predict(test_x)
    test_time_end = time.time()
    lsa_count_test_time = test_time_end - test_time_start
    lsa_count_acc = accuracy_score(pred_y, test_y)
    print('Test Accuracy : %0.3f' % lsa_count_acc)
    rmtree(cachedir)
    performance_stats_lists['accuracy'].append(lsa_count_acc)
    performance_stats_lists['training_time'].append(lsa_count_train_time)
    performance_stats_lists['test_time'].append(lsa_count_test_time)
    print('-----')
    print()
    print('---------------')

    model_names = ['TF-IDF', 'Count', 'TF-IDF+Chi2', 'Count+Chi2', 'TF-IDF+SVD', 'Count+SVD']
    print('Accuracy\tTraining Time\tTesting Time')
    for i, name in enumerate(model_names):
        print('{:.3f}\t{:.3f}\t{:.3f}'.format(performance_stats_lists['accuracy'][i],
                                              float(performance_stats_lists['training_time'][i]),
                                              float(performance_stats_lists['test_time'][i])))


if __name__ == '__main__':
    main()
