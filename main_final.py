"""
Your imports remain untouched
"""

from bs4 import BeautifulSoup
from gensim.models import Word2Vec, word2vec
from lime import lime_tabular
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords as sw
from pathlib import Path

from simpletransformers.classification import ClassificationModel, ClassificationArgs
from simpletransformers.model import TransformerModel

from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from tqdm import tqdm, tqdm_pandas, tqdm_notebook

import sys
import gensim
import lime
import logging
import math
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import random
import re
import seaborn as sns
import sklearn
import spacy
import sqlite3
import string
#From helper file
from influence_pruning import (
    batched_influence_pruning,
    approx_inverse_hessian,
)
#For the dataset pruning algorithm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
import torch
from torch.utils.data import Subset

def get_classification_results(models, classifiers, cm=False):
    classification_results = []
    for model_name, model in models:
        target = df_clean['fraudulent'].copy()
        target.columns = ['fraudulent']
        target = target.astype(int)
        if 'fraudulent' in model.columns:
            features = model.drop(['fraudulent'], axis=1)
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

        for name, classifier in classifiers:
            print('Fitting ', str(name), ' on ', str(model_name))
            model_fit = classifier.fit(X_train_clf, y_train_clf.values.ravel())
            predicted = model_fit.predict(X_test_clf)
            classification_results.append((model_name, name, pd.DataFrame(classification_report(y_test_clf, predicted, output_dict=True)).transpose(), confusion_matrix(y_test_clf, predicted), matthews_corrcoef(y_test_clf, predicted)))
            if cm:
                disp = ConfusionMatrixDisplay.from_estimator(model_fit, X_test_clf, y_test_clf, cmap=plt.cm.Blues)
                disp.ax_.set_title(str('Confusion Matrix for ' + str(model_name) + ' ' + str(name)))
    return classification_results


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    tqdm.pandas()

    # Creating a list of classifiers to be used for testing later
    classifiers = []
    classifiers.append(('LR', LogisticRegression(multi_class='ovr', max_iter=500, random_state=42)))
    classifiers.append(('SGD', SGDClassifier(random_state=42)))
    classifiers.append(('KNN', KNeighborsClassifier()))
    classifiers.append(('CART', DecisionTreeClassifier(random_state=42)))
    classifiers.append(('SVM', SVC(random_state=42)))
    classifiers.append(('RF', RandomForestClassifier(random_state=42, n_estimators=100)))
    classifiers.append(('AB', AdaBoostClassifier(random_state=42, n_estimators=100)))
    classifiers.append(('GB', GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=1.0)))


    def get_classification_results(models, classifiers, cm=False):
        classification_results = []

        for model_name, model in models:
            target = df_clean['fraudulent'].copy()
            target.columns = ["fake"]
            target = target.astype(int)
            if 'fraudulent' in model.columns:
                features = model.drop(['fraudulent'], axis=1)
            X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(features, target, test_size=0.2,
                                                                                stratify=target, random_state=42)

            for name, classifier in classifiers:
                print('Fitting ', str(name), ' on ', str(model_name))
                model_fit = classifier.fit(X_train_clf, y_train_clf.values.ravel())
                predicted = model_fit.predict(X_test_clf)
                classification_results.append((model_name, name, pd.DataFrame(
                    classification_report(y_test_clf, predicted, output_dict=True)).transpose(),
                                            confusion_matrix(y_test_clf, predicted),
                                            matthews_corrcoef(y_test_clf, predicted)))
                if cm == True:
                    disp = ConfusionMatrixDisplay.from_estimator(model_fit, X_test_clf, y_test_clf, cmap=plt.cm.Blues)
                    disp.ax_.set_title(str('Confusion Matrix for ' + str(model_name) + ' ' + str(name)))
                    plt.show()

        return (classification_results)


    """
    Data Processing
    """

    #Loading the base EMSCAD Dataset
    df = pd.read_csv('fake_job_postings.csv')

    df = df.drop(columns=['job_id'])

    #Deep Learning Pretrained models
    model_args = {
        'num_train_epochs': 5,
        'max_seq_length': 256,
        'train_batch_size': 16,
        'eval_batch_size': 64,
        'warmup_steps': 500,
        'weight_decay': 0.01,
        'logging_steps': 10,
        'learning_rate': 5e-5,
        'fp16': False,
        'no_cache': True,
        'overwrite_output_dir': True
    }
    trans_model_bert = ClassificationModel('bert', 'bert-base-cased', num_labels=4, use_cuda=False, args=model_args)
    trans_model_roberta = ClassificationModel('roberta', 'roberta-base', num_labels=4, use_cuda=False, args=model_args)
    trans_model_xlnet = ClassificationModel('xlnet', 'xlnet-base-cased', num_labels=4, use_cuda=False, args=model_args)

###Dataset Pruning Test
    #Prepare Dataset class for BERT
    class JobDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=256):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, item):
            encoded = self.tokenizer(
                self.texts[item],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
                "labels": torch.tensor(self.labels[item], dtype=torch.long)
            }
    #Tokenizer and dataset reduction for testing
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    df_pruned = df

###For subsetting the dataset
    #Split the dataset to preserve class ratio
    df_real = df[df['fraudulent'] == 0]  # Real (non-fake) cases
    df_fake = df[df['fraudulent'] == 1]  # Fake cases
    #Sample a fraction from each
    df_real_sampled = df_real.sample(frac=.05, random_state=42)
    df_fake_sampled = df_fake.sample(frac=.05, random_state=42)
    #Merge the subsets to create a new dataset
    df_pruned = pd.concat([df_real_sampled, df_fake_sampled])
    #Shuffle the dataset
    df_pruned = df_pruned.sample(frac=1, random_state=42).reset_index(drop=True)
    df_pruned['description'] = df_pruned['description'].apply(str)

    """
    #Preparing it for the JobDataset Class
    texts = df_pruned['description'].tolist()
    labels = df_pruned['fraudulent'].tolist()
    df_temp = pd.DataFrame({'description': texts, 'label': labels})
    stratified_subset = df_temp
    # Rebuild the dataset using the stratified subset
    subset_dataset = JobDataset(
        stratified_subset['description'].astype(str).tolist(), 
        stratified_subset['label'].tolist(), 
        tokenizer
    )
    loader = DataLoader(subset_dataset, batch_size=1)

    #Getting gradients
    loss_fn = nn.CrossEntropyLoss()
    model_bert = trans_model_bert.model  # Use the actual transformer model inside ClassificationModel
    model_bert.eval()
    H_inv = approx_inverse_hessian(model_bert, loader, loss_fn)

    #Influence and pruning combined, batched into sets of 100 so that memory limits can be accounted for
    selected_indices = batched_influence_pruning(model_bert, full_dataset=subset_dataset, loss_fn=loss_fn, H_inv=H_inv, batch_size=100, epsilon=.2)

    #Creating the new DataFrame
    df_pruned = df.iloc[selected_indices].reset_index(drop=True)
    #Saving the new DataFrame so this step can be skipped in the future
    df_pruned.to_csv('pruned_fake_job_postings.csv', index=False)

### End of implemented pruning
    """

    #Shuffle the dataset again
    combined_jobs_upsampled = df_pruned.sample(frac=1, random_state=42).reset_index(drop=True)
    combined_jobs_upsampled['fraudulent'].value_counts()
    combined_jobs_upsampled['description'] = combined_jobs_upsampled['description'].apply(str)

    #Load NLTK and spaCy libraries
    nltk.download('stopwords')
    nltk.download('words')

    nlp = spacy.load("en_core_web_lg")
    stopwords = sw.words('english')
    words = set(nltk.corpus.words.words())
    punctuations = string.punctuation


    # Function: Clean Text
    def process_text(docs):
        texts = []
        for doc in tqdm(docs):                       #changed tqdm(str(docs)) to tqdm(docs) to prevent nan value generation
            doc = re.sub(r'\d+\n/gm', ' ', doc)
            doc = nlp(doc, disable=['parser', 'ner'])
            tokens = [tok.lemma_.lower().strip() for tok in doc]
            tokens = [tok for tok in tokens if
                    tok in words and tok not in stopwords and tok not in punctuations and tok.isnumeric() is False]
            tokens = ' '.join(tokens)
            texts.append(tokens)
        return pd.Series(texts)


    # Creating a dataframe of cleaned job advertisements
    df_clean = combined_jobs_upsampled.copy()
    df_clean['description'] = process_text(df_clean['description'])


    """
    Bag of Words / TfIdf Feature Sets
    """

    df_pre_bow = df_clean.copy()
    df_pre_bow = df_pre_bow.reset_index(drop=True)

    # Applying the CountVectorizer and TfidfVectorizer functions
    count_vectorizer = CountVectorizer(stop_words="english")
    tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words="english", lowercase=False)

    # Fitting on the cleaned text
    vectorized_bow = count_vectorizer.fit_transform(df_pre_bow['description'])
    vectorized_tfidf = tfidf_vectorizer.fit_transform(df_pre_bow['description'])

    # Finalising the dataframe with tfidf and bow vectors
    df_bow = pd.DataFrame(vectorized_bow.todense(), columns=count_vectorizer.get_feature_names_out()) #switched to get_feature_names_out suffix for 1.0 update
    df_bow = pd.concat([df_bow, df_clean['fraudulent']], axis=1, join="inner")

    df_tfidf = pd.DataFrame(vectorized_tfidf.todense(), columns=tfidf_vectorizer.get_feature_names_out())
    df_tfidf = pd.concat([df_tfidf, df_clean['fraudulent']], axis=1, join="inner")

    models_class1 = []
    models_class1.append(('1: BOW', df_bow))
    models_class1.append(('1: TFIDF', df_tfidf))

    # Test the classification of the BOW and TFIDF Models
    get_classification_results(models_class1, classifiers)

    """
    Ruleset Features
    """

    # Creating a ruleset dataframe based on the unprocessed dataset
    df_ruleset = combined_jobs_upsampled.copy()

    ## Setting up the functions for data processing and feature extraction

    # Function: Presence of Word in String
    def is_word_in_field(docs, word):
        word_in_string = []
        for doc in tqdm(docs):
            if word.lower() in str(doc).lower():
                word_in_string.append(1)
            else:
                word_in_string.append(0)
        return pd.Series(word_in_string)


    # Function: Presence of Words in String
    def are_words_in_field(docs, words):
        contains = []
        for doc in tqdm(docs):
            has_tokens = 0
            for word in words:
                if word.lower() in str(doc).lower():
                    has_tokens = 1
            contains.append(has_tokens)
        return pd.Series(contains)


    # Feature: Number of Consecutive Punctuation
    def get_consecutive_punctuation(docs):
        counts = []
        for doc in tqdm(docs):
            count = 0
            doc = nlp(str(doc), disable=['parser', 'ner'])
            tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
            tokens = list(zip(tokens[1:], tokens))
            for tok in tokens:
                if tok[0] in punctuations and tok[1] in punctuations:
                    count += 1
            counts.append(count)
        return pd.Series(counts)


    # Function: Get POS Tag
    def get_pos_tags(text, tag):
        count = 0
        doc = nlp(text)
        for tok in doc:
            if tok.pos_ != tag:
                continue
            count += 1
        return count


    # Extract Feature: No company profile
    empty_profiles = []
    for profile in df_ruleset['company_profile']:
        if profile == ' ':
            empty_profiles.append(1)
        else:
            empty_profiles.append(0)

    df_ruleset['has_no_company_profile'] = pd.Series(empty_profiles)

    # Extract Feature: Short Company Profile
    short_profiles = []
    for desc in tqdm(df_ruleset['company_profile']):
        if len(str(desc)) < 10:
            short_profiles.append(1)
        else:
            short_profiles.append(0)

    df_ruleset['has_short_company_profile'] = pd.Series(short_profiles)

    # Extract Feature: Has No Long Company Profile
    no_long_profiles = []
    for desc in tqdm(df_ruleset['company_profile']):
        if 10 < len(str(desc)) < 100:
            no_long_profiles.append(1)
        else:
            no_long_profiles.append(0)

    df_ruleset['has_no_long_company_profile'] = pd.Series(no_long_profiles)

    # Extract Feature: Has Short Description       
    short_descriptions = []
    for desc in tqdm(df_ruleset['description']):            #description to text
        if len(str(desc)) < 10:
            short_descriptions.append(1)
        else:
            short_descriptions.append(0)

    df_ruleset['has_short_description'] = pd.Series(short_descriptions)

    # Extract Feature: Has Short Requirements
    short_requirements = []
    for desc in tqdm(df_ruleset['requirements']):
        if len(str(desc)) < 10:
            short_requirements.append(1)
        else:
            short_requirements.append(0)

    df_ruleset['has_short_requirements'] = pd.Series(short_requirements)

    # Extract Feature: Contains Spam Word
    df_ruleset['contains_spamword'] = are_words_in_field(df_ruleset['description'],
                                                        ['home', 'online', 'week', 'income', 'extra', 'cash'])

    # Extract Feature: Consecutive Punctuation
    df_ruleset['consecutive_punct'] = get_consecutive_punctuation(df_ruleset['description'])

    # Extract Feature: Has Money Symbols in Title
    df_ruleset['money_in_title'] = are_words_in_field(df_ruleset['title'],
                                                    ['$', '$$', '$$$', '€', '€€', '€€€', '£', '££', '£££'])

    # Extract Feature: Has Money Symbols in Description
    df_ruleset['money_in_description'] = are_words_in_field(df_ruleset['description'],
                                                            ['$', '$$', '$$$', '€', '€€', '€€€', '£', '££', '£££'])

    # Extract Feature: Has URL in Text
    df_ruleset['url_in_text'] = is_word_in_field(df_ruleset['description'], "url")

    # Extract Feature: Prompts for External Application
    df_ruleset['external_application'] = are_words_in_field(df_ruleset['description'],
                                                            ['apply at', 'send resume', 'send your resume', 'contact us',
                                                            'call me', 'call us'])

    # Extract Feature: Addresses Lower Education
    df_ruleset['addresses_lower_education'] = are_words_in_field(df_ruleset['description'], ['high school', 'no degree'])

    # Extract Feature: Has Incomplete Extra Attributes
    extra_cols = ['industry', 'function', 'employment_type', 'required_education']
    df_ruleset['has_incomplete_extra_attributes'] = pd.Series(df_ruleset[extra_cols].isna().any(axis=1).astype(int))

    # Extract Feature: POS Tags
    df_pos = df_ruleset.copy()
    df_pos['noun_count'] = df_pos['description'].progress_apply(lambda x: get_pos_tags(x, 'NOUN'))
    df_pos['verb_count'] = df_pos['description'].progress_apply(lambda x: get_pos_tags(x, 'VERB'))
    df_pos['adj_count'] = df_pos['description'].progress_apply(lambda x: get_pos_tags(x, 'ADJ'))
    df_pos['adv_count'] = df_pos['description'].progress_apply(lambda x: get_pos_tags(x, 'ADV'))
    df_pos['pron_count'] = df_pos['description'].progress_apply(lambda x: get_pos_tags(x, 'PRON'))

    # Finalising the Ruleset Models
    df_no_pos = df_ruleset.copy()
    df_pos.drop(['company_profile', 'requirements', 'benefits', 'title', 'department', 'salary_range', 'location',
                'employment_type', 'required_experience', 'required_education', 'industry',
                'function', 'description'], axis=1, inplace=True)
    df_no_pos.drop(['company_profile', 'requirements', 'benefits', 'title', 'department', 'salary_range', 'location',
                    'employment_type', 'required_experience', 'required_education', 'industry',
                    'function', 'description'], axis=1, inplace=True)

    print(df_no_pos.columns)

    # Feature Correlations
    df_no_pos_correlations = pd.concat([df_no_pos.drop(['fraudulent'], axis=1), pd.get_dummies(df_no_pos['fraudulent'])], axis=1)

    plt.figure(figsize=(15, 12))
    correlations = df_no_pos_correlations.corr()
    sns.heatmap(correlations, annot=True, cmap=plt.cm.Blues)
    plt.show()

    #try 2, 3 when full set
    for i in [0, 1]: 
        cor_target = (correlations[i])
        print('Top feature correlations for class ', i)
        print(cor_target[abs(cor_target) > 0.05])
        print(' ')

    # Dropping features with 1-1 correlations
    df_no_pos.drop(['telecommuting', 'has_no_company_profile'], axis=1, inplace=True)
    df_pos.drop(['telecommuting', 'has_no_company_profile'], axis=1, inplace=True)


    print(df_bow.columns)
    # Combining Ruleset+POS with Tf-Idf / BoW Models
    #df_bow = df_bow.drop(['fraudulent'], axis=1)
    #df_tfidf = df_tfidf.drop(['fraudulent'], axis=1)

    df_pos_bow = pd.concat([df_pos.reset_index(drop=True), df_bow.reset_index(drop=True)], axis=1)
    df_pos_tfidf = pd.concat([df_pos.reset_index(drop=True), df_tfidf.reset_index(drop=True)], axis=1)

    models_class2 = []
    models_class2.append(('2: Ruleset', df_no_pos))
    models_class2.append(('2: Ruleset+POS', df_pos))
    models_class2.append(('2: Ruleset+POS+BOW', df_pos_bow))
    models_class2.append(('2: Ruleset+POS+TFIDF', df_pos_tfidf))

    get_classification_results(models_class2, classifiers)

    # RF Feature Importance
    target = df_clean['fraudulent'].copy()
    target.columns = ['fraudulent']
    target = target.astype(int)
    features = df_pos.drop(['fraudulent'], axis=1)

    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(features, target, test_size=0.2,
                                                                    stratify=target, random_state=42)
    feature_names = X_train_rf.columns

    clf_rf_model = RandomForestClassifier(random_state=42).fit(X_train_rf, y_train_rf.values.ravel())

    result = permutation_importance(clf_rf_model, X_train_rf, y_train_rf, n_repeats=15, random_state=42, n_jobs=1)

    rf_importance = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)

    # Plotting Random Forest Feature Importance
    fig, ax = plt.subplots()
    rf_importance.plot.bar(ax=ax)
    ax.set_title("Feature Importances using Permutation")
    ax.set_ylabel("Mean accuracy decrease")
    # fig.tight_layout()
    plt.show()

    rf_importance

    """
    Word2Vec Embeddings
    """

    logging.basicConfig(level=logging.INFO)
    nltk.download('punkt')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def sentence_to_words(text):
        snt_words = re.sub("[^a-zA-Z]", " ", text)
        snt_words_split = snt_words.lower().split()
        return snt_words_split


    def doc_to_sentences(doc, tokenizer):
        sentences = []
        tokenized_sentences = tokenizer.tokenize(doc.strip())
        for sentence in tokenized_sentences:
            if len(sentence) > 0:
                sentences.append(sentence_to_words(sentence))
        return sentences


    def gen_feature_vec(doc, embedding, num_features):
        vec = np.zeros(num_features, dtype="float32")
        word_count = 0

        word_index2word = set(embedding.wv.index_to_key)
        for word in doc:
            if word not in word_index2word:
                continue
            word_count += 1
            vec = np.add(vec, embedding.wv[word])

        vec = np.divide(vec, word_count)
        return vec


    def avg_feature_vec(docs, embeddings, num_features):
        feature_vecs = np.zeros((len(docs), num_features), dtype='float32')

        i = 0
        for doc in docs:
            feature_vecs[i] = gen_feature_vec(doc, embeddings, num_features)
            i += 1

        return feature_vecs

    # Creating the training and testing sets for the embeddings
    target_embed = combined_jobs_upsampled['fraudulent'].copy()
    target_embed.columns = ['fraudulent']
    target_embed = target_embed.astype(int)
    features_embed = combined_jobs_upsampled['description']
    X_train_embed, X_test_embed, y_train_embed, y_test_embed = train_test_split(features_embed, target_embed, test_size=0.2,
                                                                                stratify=target_embed, random_state=42)

    train_data_embed = pd.DataFrame()
    test_data_embed = pd.DataFrame()

    train_data_embed['description'] = X_train_embed
    train_data_embed['labels'] = y_train_embed

    test_data_embed['description'] = X_test_embed
    test_data_embed['labels'] = y_test_embed

    train_sentences = []  # Initialize an empty list of sentences
    for doc in train_data_embed['description']:
        train_sentences += doc_to_sentences(doc, tokenizer)

    # Training the word2vec model
    w2v_model = word2vec.Word2Vec(train_sentences, workers=3,
                                vector_size=300, min_count=40,
                                window=10, sample=1e-3)

    train_words_processed = process_text(train_data_embed['description'])
    test_words_processed = process_text(test_data_embed['description'])

    cleaned_train_text = []
    cleaned_test_text = []

    for doc in train_words_processed:
        cleaned_train_text.append(sentence_to_words(doc))

    for doc in test_words_processed:
        cleaned_test_text.append(sentence_to_words(doc))

    df_embed_train = pd.DataFrame(avg_feature_vec(cleaned_train_text, w2v_model, 300)).fillna(0)
    df_embed_test = pd.DataFrame(avg_feature_vec(cleaned_test_text, w2v_model, 300)).fillna(0)

    df_embed_train.head

    # Combining Ruleset with Embeddings
    df_embed_ruleset_train = pd.concat([df_embed_train.reset_index(drop=True), X_train_rf.reset_index(drop=True)],
                                    axis=1).fillna(0)
    df_embed_ruleset_test = pd.concat([df_embed_test.reset_index(drop=True), X_test_rf.reset_index(drop=True)],
                                    axis=1).fillna(0)

    #added string typing to fix scikit error in next chunk
    df_embed_ruleset_train.columns = df_embed_ruleset_train.columns.astype(str)
    df_embed_ruleset_test.columns = df_embed_ruleset_test.columns.astype(str)



    #Testing pruning
    """

    # Evaluating the Word2Vec and CombinedWord2vec Models
    for name, model in classifiers:
        model_fit = model.fit(df_embed_train, y_train_embed.values.ravel())
        predicted = model_fit.predict(df_embed_test)
        print("Classification report of " + str(name) + " tested on Embeddings test set:")
        print(classification_report(y_test_embed, predicted, digits=3))
        print("")
        print(matthews_corrcoef(y_test_embed, predicted))

    for name, model in classifiers:
        model_fit = model.fit(df_embed_ruleset_train, y_train_embed.values.ravel())
        predicted = model_fit.predict(df_embed_ruleset_test)
        print("Classification report of " + str(name) + " tested on Embeddings+Ruleset test set:")
        print(classification_report(y_test_embed, predicted, digits=6))
        print("")
        print(matthews_corrcoef(y_test_embed, predicted))

    # Hyperparameter tuning
    grid_param = {
        'n_estimators': [90, 110, 130, 150],
        'criterion': ['gini', 'entropy'],
        'max_depth': range(2, 20, 5),
        'min_samples_leaf': range(1, 10, 2),
        'min_samples_split': range(2, 10, 2),
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=grid_param, cv=5, n_jobs=-1,
                            verbose=3)
    grid_search.fit(df_embed_ruleset_train, y_train_embed.values.ravel())

    grid_search.best_params_

    """

    # RF Hyperparameter Tuning: Ruleset + Embeddings
    clf_rf_combined = RandomForestClassifier(random_state=42, criterion='entropy',
                                            max_depth=17, max_features='sqrt',
                                            min_samples_leaf=1, min_samples_split=2,
                                            n_estimators=90).fit(df_embed_ruleset_train, y_train_embed.values.ravel())

    pred_clf_combined = clf_rf_combined.predict(df_embed_ruleset_test)
    print(confusion_matrix(y_test_embed, pred_clf_combined))
    print(classification_report(y_true=y_test_embed, y_pred=pred_clf_combined))


    """
    Transformer-based Model
    """

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    data_tf = pd.DataFrame()
    data_tf['description'] = combined_jobs_upsampled['description']
    data_tf['value'] = combined_jobs_upsampled['fraudulent']

    target_trans = data_tf['value'].copy()
    features_trans = data_tf['description'].copy()
    X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(features_trans, target_trans, test_size=0.2,
                                                                                stratify=target_trans, random_state=42)

    train_data_tf = pd.DataFrame()
    test_data_tf = pd.DataFrame()

    train_data_tf['description'] = X_train_trans
    train_data_tf['labels'] = y_train_trans

    test_data_tf['description'] = X_test_trans
    test_data_tf['labels'] = y_test_trans


    print("Before Bert")

    # Training and Evaluating BERT

    
    trans_model_bert.train_model(train_data_tf)
    result_bert, model_outputs_bert, wrong_predictions_bert = trans_model_bert.eval_model(test_data_tf,
                                                                                        acc=sklearn.metrics.accuracy_score,
                                                                                        cm=sklearn.metrics.confusion_matrix)


    predictions_bert, raw_outputs_bert = trans_model_bert.predict(list(X_test_trans))
    pd.DataFrame(classification_report(y_pred=predictions_bert, y_true=y_test_trans, output_dict=True)).transpose()
    disp = ConfusionMatrixDisplay(confusion_matrix=result_bert['cm'],
                                display_labels=np.asarray([0, 1]))

    disp = disp.plot(cmap=plt.cm.Blues, values_format='')
    plt.show()


    # Training and Evaluating RoBERTa
    trans_model_roberta.train_model(train_data_tf)
    result_roberta, model_outputs_roberta, wrong_predictions_roberta = trans_model_roberta.eval_model(test_data_tf,
                                                                                                    acc=sklearn.metrics.accuracy_score,
                                                                                                    cm=sklearn.metrics.confusion_matrix)

    predictions_roberta, raw_outputs_roberta = trans_model_roberta.predict(list(X_test_trans))
    pd.DataFrame(classification_report(y_pred=predictions_roberta, y_true=y_test_trans, output_dict=True)).transpose()

    disp = ConfusionMatrixDisplay(confusion_matrix=result_roberta['cm'],
                                display_labels=np.asarray([0, 1]))

    disp = disp.plot(cmap=plt.cm.Blues, values_format='')
    plt.show()

    # Training and Evaluating XLNet
    trans_model_xlnet.train_model(train_data_tf)
    result_xlnet, model_outputs_xlnet, wrong_predictions_xlnet = trans_model_xlnet.eval_model(test_data_tf,
                                                                                            acc=sklearn.metrics.accuracy_score,
                                                                                            cm=sklearn.metrics.confusion_matrix)

    predictions_xlnet, raw_outputs_xlnet = trans_model_xlnet.predict(list(X_test_trans))
    pd.DataFrame(classification_report(y_pred=predictions_xlnet, y_true=y_test_trans, output_dict=True)).transpose()

    disp = ConfusionMatrixDisplay(confusion_matrix=result_xlnet['cm'],
                                display_labels=np.asarray([0, 1]))

    disp = disp.plot(cmap=plt.cm.Blues, values_format='')
    plt.show()

    print("Classification report for BERT:")
    print(pd.DataFrame(classification_report(y_pred=predictions_bert, y_true=y_test_trans, output_dict=True)).transpose())
    print(" ")
    print("Classification report for RoBERTa:")
    print(
        pd.DataFrame(classification_report(y_pred=predictions_roberta, y_true=y_test_trans, output_dict=True)).transpose())
    print(" ")
    print("Classification report for XLNet:")
    print(pd.DataFrame(classification_report(y_pred=predictions_xlnet, y_true=y_test_trans, output_dict=True)).transpose())
