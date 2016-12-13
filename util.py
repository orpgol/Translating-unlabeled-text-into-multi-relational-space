# Copyright 2016 Mandiant, A FireEye Company
# Authors: Brian Jones
# License: Apache 2.0

''' Utility functions for "Relational Learning with TensorFlow" tutorial '''

import numpy as np
import pandas as pd


def df_to_idx_array(df):
    '''Converts a Pandas DataFrame containing relationship triples
       into a numpy index array.

    Args:
        df: Pandas DataFrame with columns 'head', 'rel', and 'tail'. These
            columns must be Categorical. See make_categorical().

    Returns:
        A (N x 3) numpy integer index array built from the column Categorical
            codes.
    '''
    idx_array = np.zeros((len(df),3), dtype=np.int)
    idx_array[:,0] = df['head'].cat.codes
    idx_array[:,1] = df['rel'].cat.codes
    idx_array[:,2] = df['tail'].cat.codes
    return idx_array


def make_categorical(df, field_sets):
    '''Make DataFrame columns Categorical so that they can be converted to
       index arrays for feeding into TensorFlow models.

    Args:
        df: Pandas DataFrame with columns 'head', 'rel', and 'tail'
        field_sets: A tuples containing the item category sets: (head_set,
            rel_set, tail_set). Note that head_set and tail_set can
            be the same if the model embeds all entities into a common
            space.

    Returns:
        A new Pandas DataFrame where the 'head', 'rel', and 'tail' columns have
        been made Caetgorical using the supplied field_sets.
    '''
    head_set, rel_set, tail_set = field_sets
    result = pd.DataFrame()
    result['head'] = pd.Categorical(df['head'].values, categories=head_set)
    result['rel'] = pd.Categorical(df['rel'].values, categories=rel_set)
    result['tail'] = pd.Categorical(df['tail'].values, categories=tail_set)
    if 'truth_flag' in df:
        result['truth_flag'] = df['truth_flag']
    return result, df_to_idx_array(result)


def corrupt(triple, field_replacements, forbidden_set,
            rng, fields=[0,2], max_tries=1000):
    ''' Produces a corrupted negative triple for the supplied positive triple
    using rejection sampling. Only a single field (from one in the fields
    argument) is changed.

    Args:
        triple: A tuple or list with 3 entries: (head, rel, tail)

        field_replacements: A tuple of array-like: (head entities, relationships,
            tail entities), each containing the (unique) items to use as
            replacements for the corruption

        forbidden_set: A set of triples (typically all known true triples)
            that we should not accidentally create when generating corrupted
            negatives.

        rng: Numpy RandomState object

        fields: The fields that can be replaced in the triple. Default is
            [0,2] which corresponds to the head and tail entries. [0,1,2]
            would randomly replace any of the three entries.

        max_tries: The maximum number of random corruption attempts before
            giving up and throwing an exception. A corruption attempt can fail
            if the sampled negative is a triple found in forbidden_set.

    Returns:
        A corrupted tuple (head, rel, tail) where one entry is different
        than the triple passed in.
    '''
    collision = False
    for _ in range(max_tries):
        field = rng.choice(fields)
        replacements = field_replacements[field]
        corrupted = list(triple)
        corrupted[field] = replacements[rng.randint(len(replacements))]
        collision = (tuple(corrupted) in forbidden_set)
        if not collision:
            break
    if collision:
        raise Exception('Failed to sample a corruption for {} after {} tries'.format(triple, max_tries))
    return corrupted


def create_tf_pairs(true_df, all_true_df, rng):
    '''Creates a DataFrame with constrastive positive/negative pairs given
       true triples to constrast and set of "all known" true triples in order
       to avoid accidentally sampling a negative from this set.

    Args:
        true_df: Pandas DataFrame containing true triples to contrast.
            It must contain columns 'head', 'rel', and 'tail'. One
            random negative will be created for each.
        all_true_df: Pandas DataFrame containing "all known" true triples.
            This will be used to to avoid randomly generating negatives
            that happen to be true but were not in true_df.
        rng: A Numpy RandomState object

    Returns:
        A new Pandas DataFrame with alternating pos/neg pairs. If true_df
        contains rows [p1, p2, ..., pN], then this will contain 2N rows in the
        form [p1, n1, p2, n2, ..., pN, nN].
    '''
    all_true_tuples = set(all_true_df.itertuples(index=False))
    replacements = (list(set(true_df['head'])), [], list(set(true_df['tail'])))
    result = []
    for triple in true_df.itertuples(index=False):
        corruption = corrupt(triple, replacements, all_true_tuples, rng)
        result.append(triple)
        result.append(corruption)
    result = pd.DataFrame(result, columns=['head', 'rel', 'tail'])
    result['truth_flag'] = np.tile([True, False], len(true_df))
    return result


def threshold_and_eval(test_df, test_scores, val_df, val_scores):
    ''' Test set evaluation protocol from:
        Socher, Richard, et al. "Reasoning with neural tensor networks for
        knowledge base completion." Advances in Neural Information Processing
        Systems. 2013.

    Finds model output thresholds using val_df to create a binary
    classifier, and then measures classification accuracy on the test
    set scores using these thresholds. A different threshold is found
    for each relationship type. All Dataframes must have a 'rel' column.

    Args:
        test_df: Pandas DataFrame containing the test triples
        test_scores: A numpy array of test set scores, one for each triple
            in test_df
        val_df: A Pandas DataFrame containing the validation triples
        test_scores: A numpy array of validation set scores, one for each triple
            in val_df

    Returns:
        A tuple containing (accuracy, test_predictions, test_scores, threshold_map)
            accuracy: the overall classification accuracy on the test set
            test_predictions: True/False output for test set
            test_scores: Test set scores
            threshold_map: A dict containing the per-relationship thresholds
                found on the validation set, e.g. {'_has_part': 0.562}
    '''
    def find_thresh(df, scores):
        ''' find threshold that maximizes accuracy on validation set '''
        #print(df.shape, scores.shape)
        sorted_scores = sorted(scores)
        best_score, best_thresh = -np.inf, -np.inf
        for i in range(len(sorted_scores)-1):
            thresh = (sorted_scores[i] + sorted_scores[i+1]) / 2.0
            predictions = (scores > thresh)
            correct = np.sum(predictions == df['truth_flag'])
            if correct >= best_score:
                best_score, best_thresh = correct, thresh
        return best_thresh
    threshold_map = {}
    for relationship in set(val_df['rel']):
        mask = np.array(val_df['rel'] == relationship)
        threshold_map[relationship] = find_thresh(val_df.loc[mask], val_scores[mask])
    test_entry_thresholds = np.array([threshold_map[r] for r in test_df['rel']])
    test_predictions = (test_scores > test_entry_thresholds)
    accuracy = np.sum(test_predictions == test_df['truth_flag']) / len(test_predictions)
    return accuracy, test_predictions, test_scores, threshold_map


def model_threshold_and_eval(model, test_df, val_df):
    ''' See threshold_and_eval(). This is the same except that the supplied
    model will be used to generate the test_scores and val_scores.

    Args:
        model: A trained relational learning model whose predict() will be
            called on index arrays generated from test_df and val_df
        test_df: Pandas DataFrame containing the test triples
        val_df: A Pandas DataFrame containing the validation triples

    Returns:
        A tuple containing (accuracy, test_predictions, test_scores, threshold_map)
            accuracy: the overall classification accuracy on the test set
            test_predictions: True/False output for test set
            test_scores: Test set scores
            threshold_map: A dict containing the per-relationship thresholds
                found on the validation set, e.g. {'_has_part': 0.562}
    '''
    val_scores = model.predict(df_to_idx_array(val_df))
    test_scores = model.predict(df_to_idx_array(test_df))
    return threshold_and_eval(test_df, test_scores, val_df, val_scores)


def pair_ranking_accuracy(model_output):
    ''' Pair ranking accuracy. This only works when model_output comes from
    alternating positive/negative pairs: [pos,neg,pos,neg,...,pos,neg]

    Returns:
        The fraction of pairs for which the positive example is scored higher
        than the negative example
    '''
    output_pairs = np.reshape(model_output, [-1,2])
    correct = np.sum(output_pairs[:,0] > output_pairs[:,1])
    return float(correct) / len(output_pairs)


def model_pair_ranking_accuracy(model, data):
    ''' See pair_ranking_accuracy(), this simply calls model.predict(data) to
    generate model_output

    Returns:
        The fraction of pairs for which the positive example is scored higher
        than the negative example
    '''
    return pair_ranking_accuracy(model.predict(data))


class ContrastiveTrainingProvider(object):
    ''' Provides mini-batches for stochastic gradient descent by augmenting
    a set of positive training triples with random contrastive negative samples.

    Args:
        train: A 2D numpy array with positive training triples in its rows
        batch_pos_cnt: Number of positive examples to use in each mini-batch
        separate_head_tail: If True, head and tail corruptions are sampled
            from entity sets limited to those found in the respective location.
            If False, head and tail replacements are sampled from the set of
            all entities, regardless of location.
        rng: (optional) A NumPy RandomState object

    TODO: Allow a variable number of negative examples per positive. Right
    now this class always provides a single negative per positive, generating
    pairs: [pos, neg, pos, neg, ...]
    '''

    def __init__(self, train, batch_pos_cnt=50,
                 separate_head_tail=False, rng=None):
        self.train = train
        self.batch_pos_cnt = batch_pos_cnt
        self.separate_head_tail = separate_head_tail
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.num_examples = len(train)
        self.epochs_completed = 0
        self.index_in_epoch = 0
        # store set of training tuples for quickly checking negatives
        self.triples_set = set(tuple(t) for t in train)
        # replacement entities
        if separate_head_tail:
            head_replacements = list(set(train[:,0]))
            tail_replacements = list(set(train[:,2]))
        else:
            all_entities = set(train[:,0]).union(train[:,2])
            head_replacements = tail_replacements = list(all_entities)
        self.field_replacements = [head_replacements,
                                   list(set(train[:,1])),
                                   tail_replacements]
        self._shuffle_data()

    def _shuffle_data(self):
        self.rng.shuffle(self.train)

    def next_batch(self):
        '''
        Returns:
            A tuple (batch_triples, batch_labels):
            batch_triples: Bx3 numpy array of triples, where B=2*batch_pos_cnt
            batch_labels: numpy array with 0/1 labels for each row in
                batch_triples
            Each positive is followed by a constrasting negative, so batch_labels
            will alternate: [1, 0, 1, 0, ..., 1, 0]
        '''
        start = self.index_in_epoch
        self.index_in_epoch += self.batch_pos_cnt
        if self.index_in_epoch > self.num_examples:
            # Finished epoch, shuffle data
            self.epochs_completed += 1
            self.index_in_epoch = self.batch_pos_cnt
            start = 0
            self._shuffle_data()
        end = self.index_in_epoch
        batch_triples = []
        batch_labels = []
        for positive in self.train[start:end]:
            batch_triples.append(positive)
            batch_labels.append(1.0)
            negative = corrupt(positive, self.field_replacements, self.triples_set, self.rng)
            batch_triples.append(negative)
            batch_labels.append(0.0)
        batch_triples = np.vstack(batch_triples)
        batch_labels = np.array(batch_labels)
        return batch_triples, batch_labels
