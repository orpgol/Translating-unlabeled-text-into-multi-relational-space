# Copyright 2016 Mandiant, A FireEye Company
# Authors: Brian Jones
# License: Apache 2.0

''' Model classes for "Relational Learning with TensorFlow" tutorial '''

import numpy as np
import tensorflow as tf

from util import ContrastiveTrainingProvider


def least_squares_objective(output, target, add_bias=True):
    ''' Creates final model output and loss for least squares objective

    Args:
        output: Model output
        target: Training target placeholder
        add_bias: If True, a bias Variable will be added to the output

    Returns:
        tuple (final output, loss)
    '''
    y = output
    if add_bias:
        bias = tf.Variable([0.0])
        y = output + bias
    loss = tf.reduce_sum(tf.square(y - target))
    return y, loss


def logistic_objective(output, target, add_bias=True):
    ''' Creates final model output and loss for logistic objective

    Args:
        output: Model output
        target: Training target placeholder
        add_bias: If True, a bias Variable will be added to the output

    Returns:
        tuple (final output, loss)
    '''
    y = output
    if add_bias:
        bias = tf.Variable([0.0])
        y = output + bias
    sig_y = tf.clip_by_value(tf.sigmoid(y), 0.001, 0.999) # avoid NaNs
    loss = -tf.reduce_sum(target*tf.log(sig_y) + (1-target)*tf.log(1-sig_y))
    return sig_y, loss


def ranking_margin_objective(output, margin=1.0):
    ''' Create final model output and loss for pairwise ranking margin objective

    Loss for single pair (f(p), f(n)) = [margin - f(p) + f(n)]+
    This only works when given model output on alternating positive/negative
    pairs: [pos,neg,pos,neg,...]. TODO: check target placeholder
    at runtime to make sure this is the case?

    Args:
        output: Model output
        margin: The margin value for the pairwise hinge loss

    Returns:
        tuple (final output, loss)
    '''
    y_pairs = tf.reshape(output, [-1,2]) # fold: 1 x n -> [n/2 x 2]
    pos_scores, neg_scores = tf.split(1, 2, y_pairs) # separate pairs
    hinge_losses = tf.nn.relu(margin - pos_scores + neg_scores)
    total_hinge_loss = tf.reduce_sum(hinge_losses)
    return output, total_hinge_loss


def sparse_maxnorm_update(var_matrix, indices, maxnorm=1.0):
    '''Sparse update operation that ensures selected rows in var_matrix
       do not have a Euclidean norm greater than maxnorm. Rows that exceed
       it are scaled to length.

    Args:
        var_matrix: 2D mutable tensor (Variable) to operate on
        indices: 1D tensor with the row indices to constrain
        maxnorm: the maximum Euclidean norm

    Returns:
        An operation that will update var_matrix when run in a Session
    '''
    selected_rows = tf.nn.embedding_lookup(var_matrix, indices)
    row_norms = tf.sqrt(tf.reduce_sum(tf.square(selected_rows), 1))
    scaling = maxnorm / tf.maximum(row_norms, maxnorm)
    scaled = selected_rows * tf.expand_dims(scaling, 1)
    return tf.scatter_update(var_matrix, indices, scaled)


def dense_maxnorm_update(var_matrix, maxnorm=1.0):
    '''Dense update operation that ensures all rows in var_matrix
       do not have a Euclidean norm greater than maxnorm. Rows that exceed
       it are scaled to length.

    Args:
        var_matrix: 2D mutable tensor (Variable) to operate on
        maxnorm: the maximum Euclidean norm

    Returns:
        An operation that will update var_matrix when run in a Session
    '''
    row_norms = tf.sqrt(tf.reduce_sum(tf.square(var_matrix), 1))
    scaling = maxnorm / tf.maximum(row_norms, maxnorm)
    scaled = var_matrix * tf.expand_dims(scaling, 1)
    return tf.assign(var_matrix, scaled)


def dense_maxnorm(var_matrix, maxnorm=1.0):
    '''Similar to dense_maxnorm_update(), except this returns a new Tensor
       instead of an operation that modifies var_matrix.

    Args:
        var_matrix: 2D tensor (Variable)
        maxnorm: the maximum Euclidean norm

    Returns:
        A new tensor where all rows have been scaled as necessary
    '''
    axis_norms = tf.sqrt(tf.reduce_sum(tf.square(var_matrix), 1))
    scaling = maxnorm / tf.maximum(axis_norms, maxnorm)
    return var_matrix * tf.expand_dims(scaling, 1)


class BaseModel(object):
    ''' Base class for embedding-based relational learning models that use
        maxnorm regularization. Subclasses must implement _create_model() and
        populate self.train_step, and can optionally populate self.post_step for
        post-processing.

    Note: When model_type is 'ranking_margin', the mini-batch provider returned
    by _create_batch_provider() must provide instances in alternating
    pos/neg pairs: [pos, neg, pos, neg, ...]. This is satisfied when using
    ContrastiveTrainingProvider; be careful if you use a different one.

    Args:
        embedding_size: Embedding vector length
        maxnorm: Maximum Euclidean norm for embedding vectors
        batch_pos_cnt: Number of positive examples to use in each mini-batch
        max_iter: Maximum number of optimization iterations to perform
        model_type: Possible values:
            'least_squares': squared loss on 0/1 targets
            'logistic': sigmoid link function, crossent loss on 0/1 targets
            'ranking_margin': ranking margin on pos/neg pairs
        add_bias: If True, a bias Variable will be added to the output for
            least_squares and logistic models.
        opt: An optimizer object to use. If None, the default optimizer is
            tf.train.AdagradOptimizer(1.0)

    TODO: add support for other regularizers like L2
    '''

    def __init__(self, embedding_size, maxnorm=1.0,
                 batch_pos_cnt=100, max_iter=1000,
                 model_type='least_squares', add_bias=True,
                 opt=None):
        self.embedding_size = embedding_size
        self.maxnorm = maxnorm
        self.batch_pos_cnt = batch_pos_cnt
        self.max_iter = max_iter
        self.model_type = model_type
        self.add_bias = add_bias
        if opt is None:
            opt = tf.train.AdagradOptimizer(1.0)
        self.opt = opt
        self.sess = None
        self.train_step = None
        self.post_step = None
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.head_input = tf.placeholder(tf.int32, shape=[None])
            self.rel_input = tf.placeholder(tf.int32, shape=[None])
            self.tail_input = tf.placeholder(tf.int32, shape=[None])
            self.target = tf.placeholder(tf.float32, shape=[None])

    def _create_model(self, train_triples):
        ''' Subclasses must build Graph and set self.train_step '''
        raise Exception('subclass must implement')

    def _create_batch_provider(self, train_triples):
        ''' Default implementation '''
        return ContrastiveTrainingProvider(train_triples, self.batch_pos_cnt)

    def _create_output_and_loss(self, raw_output):
        if self.model_type == 'least_squares':
            return least_squares_objective(raw_output, self.target, self.add_bias)
        elif self.model_type == 'logistic':
            return logistic_objective(raw_output, self.target, self.add_bias)
        elif self.model_type == 'ranking_margin':
            return ranking_margin_objective(raw_output, 1.0)
        else:
            raise Exception('Unknown model_type')

    def _norm_constraint_op(self, var_matrix, row_indices, maxnorm):
        '''
        Args:
            var_matrix: A 2D Tensor holding the vectors to constrain (in rows)
            row_indices: The rows in var_tensor that are being considered for
                constraint application (typically embedding vectors for
                entities observed for a minibatch of training data). These
                will be used for a sparse variable update operation if the
                chosen optimizer only modified these entries. Otherwise
                a dense operation is used and row_indices are ignored.
            maxnorm: The maximum Euclidean norm for the rows in var_tensor

        Returns:
            An operation which will apply the constraints when run in a Session
        '''
        # Currently, TF optimizers do not update variables with zero gradient
        # except AdamOptimizer
        if isinstance(self.opt, tf.train.AdamOptimizer):
            return dense_maxnorm_update(var_matrix, maxnorm)
        else:
            return sparse_maxnorm_update(var_matrix, row_indices, maxnorm)

    def embeddings(self):
        ''' Subclass should override this if it uses different embedding
            variables

        Returns:
            A list of pairs: [(embedding name, embedding 2D Tensor)]
        '''
        return [('entity', self.entity_embedding_vars),
                ('rel', self.rel_embedding_vars)]

    def create_feed_dict(self, triples, labels=None, training=False):
        ''' Create a TensorFlow feed dict for relationship triples

        Args:
            triples: A numpy integer array of relationship triples, where each
                row contains [head idx, relationship idx, tail idx]
            labels: (optional) A label array for triples
            training: (optional) A flag indicating whether the feed dict is
                for training or test purposes. Useful for things like
                dropout where a dropout_probability variable is set differently
                in the two contexts.
        '''
        feed_dict = {self.head_input: triples[:, 0],
                     self.rel_input: triples[:, 1],
                     self.tail_input: triples[:, 2]}
        if labels is not None:
            feed_dict[self.target] = labels
        return feed_dict

    def close(self):
        ''' Closes the TensorFlow Session object '''
        self.sess.close();

    def fit(self, train_triples, step_callback=None):
        ''' Trains the model on relationship triples

        Args:
            train_triples: A numpy integer array of relationship triples, where
                each row of contains [head idx, relationship idx, tail idx]
            step_callback: (optional) A function that will be called before each
                optimization step, step_callback(iteration, feed_dict)
        '''
        if self.sess is not None:
            self.sess.close()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self._create_model(train_triples)
            self.sess.run(tf.initialize_all_variables())
        batch_provider = self._create_batch_provider(train_triples)
        for i in range(self.max_iter):
            batch_triples, batch_labels = batch_provider.next_batch()
            feed_dict = self.create_feed_dict(batch_triples, batch_labels, training=True)
            if step_callback:
                keep_going = step_callback(i, feed_dict)
                if not keep_going:
                    break
            self.sess.run(self.train_step, feed_dict)
            if self.post_step is not None:
                self.sess.run(self.post_step, feed_dict)

    def predict(self, triples):
        ''' Runs a trained model on the supplied relationship triples. fit()
            must be called before calling this function.

        Args:
            triples: A numpy integer array of relationship triples, where each
                row of contains [head idx, relationship idx, tail idx]
        '''
        feed_dict = self.create_feed_dict(triples, training=False)
        return self.sess.run(self.output, feed_dict=feed_dict)


class Contrastive_CP(BaseModel):
    ''' Model with a scoring function based on CANDECOMP/PARAFAC tensor
        decomposition. Optimization differs, however, in the use of maxnorm
        regularization and contrastive negative sampling.

    Score for (head i, rel k, tail j) triple is:  h_i^T * diag(r_k) * t_j,
    where h_i and t_j are embedding vectors for the head and tail entities,
    and r_k is an embedding vector for the relationship type.

    Args:
        embedding_size: Embedding vector length
        maxnorm: Maximum Euclidean norm for embedding vectors
        batch_pos_cnt: Number of positive examples to use in each mini-batch
        max_iter: Maximum number of optimization iterations to perform
        model_type: Possible values:
            'least_squares': squared loss on 0/1 targets
            'logistic': sigmoid link function, crossent loss on 0/1 targets
            'ranking_margin': ranking margin on pos/neg pairs
        add_bias: If True, a bias Variable will be added to the output for
            least_squares and logistic models.
        opt: An optimizer object to use. If None, the default optimizer is
            tf.train.AdagradOptimizer(1.0)

    References:
        Kolda, Tamara G., and Brett W. Bader. "Tensor decompositions and
        applications." SIAM review 51.3 (2009): 455-500.
    '''

    def _create_model(self, train_triples):
        # Count unique items to determine embedding matrix sizes
        head_cnt = len(set(train_triples[:,0]))
        rel_cnt = len(set(train_triples[:,1]))
        tail_cnt = len(set(train_triples[:,2]))
        init_sd = 1.0 / np.sqrt(self.embedding_size)
        # Embedding matrices for entities and relationship types
        head_init = tf.truncated_normal([head_cnt, self.embedding_size], stddev=init_sd)
        rel_init = tf.truncated_normal([rel_cnt, self.embedding_size], stddev=init_sd)
        tail_init = tf.truncated_normal([tail_cnt, self.embedding_size], stddev=init_sd)
        if self.maxnorm is not None:
            # Ensure maxnorm constraints are initially satisfied
            head_init = dense_maxnorm(head_init, self.maxnorm)
            rel_init = dense_maxnorm(rel_init, self.maxnorm)
            tail_init = dense_maxnorm(tail_init, self.maxnorm)
        self.head_embedding_vars = tf.Variable(head_init)
        self.rel_embedding_vars = tf.Variable(rel_init)
        self.tail_embedding_vars = tf.Variable(tail_init)
        # Embedding layer for each (head, rel, tail) triple being fed in as input
        head_embed = tf.nn.embedding_lookup(self.head_embedding_vars, self.head_input)
        rel_embed = tf.nn.embedding_lookup(self.rel_embedding_vars, self.rel_input)
        tail_embed = tf.nn.embedding_lookup(self.tail_embedding_vars, self.tail_input)
        # Model output
        raw_output = tf.reduce_sum(tf.mul(tf.mul(head_embed, rel_embed), tail_embed), 1)
        self.output, self.loss = self._create_output_and_loss(raw_output)
        # Optimization
        self.train_step = self.opt.minimize(self.loss)
        if self.maxnorm is not None:
            # Post-processing to limit embedding vars to L2 ball
            head_constraint = self._norm_constraint_op(self.head_embedding_vars,
                                                       tf.unique(self.head_input)[0],
                                                       self.maxnorm)
            rel_constraint = self._norm_constraint_op(self.rel_embedding_vars,
                                                      tf.unique(self.rel_input)[0],
                                                      self.maxnorm)
            tail_constraint = self._norm_constraint_op(self.tail_embedding_vars,
                                                       tf.unique(self.tail_input)[0],
                                                       self.maxnorm)
            self.post_step = [head_constraint, rel_constraint, tail_constraint]

    def _create_batch_provider(self, train):
        # CP treats head and tail entities separately
        return ContrastiveTrainingProvider(train,
                                           self.batch_pos_cnt,
                                           separate_head_tail=True)

    def embeddings(self):
        '''
        Returns:
            A list of pairs: [(embedding name, embedding 2D Tensor)]
        '''
        return [('head', self.head_embedding_vars),
                ('tail', self.head_embedding_vars),
                ('rel', self.rel_embedding_vars)]


class Bilinear(BaseModel):
    ''' Model with a scoring function based on the bilinear formulation of
        RESCAL. Optimization differs, however, in the use of maxnorm
        regularization and contrastive negative sampling.

    Score for (head i, rel k, tail j) triple is:  e_i^T * R_k * e_j
    where e_i and e_j are D-dimensional embedding vectors for the head and tail
    entities, and R_k is a (D x D) matrix for the relationship type
    acting as a bilinear operator.

    Args:
        embedding_size: Embedding vector length
        maxnorm: Maximum Euclidean norm for embedding vectors
        rel_maxnorm_mult: Multiplier for the maxnorm threshold used for
            relationship embeddings. Example: If maxnorm=2.0 and
            rel_maxnorm_mult=4.0, then the maxnorm constrain for relationships
            will be 2.0 * 4.0 = 8.0.
        batch_pos_cnt: Number of positive examples to use in each mini-batch
        max_iter: Maximum number of optimization iterations to perform
        model_type: Possible values:
            'least_squares': squared loss on 0/1 targets
            'logistic': sigmoid link function, crossent loss on 0/1 targets
            'ranking_margin': ranking margin on pos/neg pairs
        add_bias: If True, a bias Variable will be added to the output for
            least_squares and logistic models.
        opt: An optimizer object to use. If None, the default optimizer is
            tf.train.AdagradOptimizer(1.0)

    References:
        Nickel, Maximilian, Volker Tresp, and Hans-Peter Kriegel. "A three-way
        model for collective learning on multi-relational data." Proceedings of
        the 28th international conference on machine learning (ICML-11). 2011.
    '''

    def __init__(self, embedding_size, maxnorm=1.0, rel_maxnorm_mult=3.0,
                 batch_pos_cnt=100, max_iter=1000,
                 model_type='least_squares', add_bias=True, opt=None):
        super(Bilinear, self).__init__(
            embedding_size=embedding_size,
            maxnorm=maxnorm,
            batch_pos_cnt=batch_pos_cnt,
            max_iter=max_iter,
            model_type=model_type,
            opt=opt)
        self.rel_maxnorm_mult = rel_maxnorm_mult

    def _create_model(self, train_triples):
        # Count unique items to determine embedding matrix sizes
        entity_cnt = len(set(train_triples[:,0]).union(train_triples[:,2]))
        rel_cnt = len(set(train_triples[:,1]))
        init_sd = 1.0 / np.sqrt(self.embedding_size)
        # Embedding variables for all entities and relationship types
        entity_embedding_shape = [entity_cnt, self.embedding_size]
        # Relationship embeddings will be stored in flattened format to make
        # applying maxnorm constraints easier
        rel_embedding_shape = [rel_cnt, self.embedding_size * self.embedding_size]
        entity_init = tf.truncated_normal(entity_embedding_shape, stddev=init_sd)
        rel_init = tf.truncated_normal(rel_embedding_shape, stddev=init_sd)
        if self.maxnorm is not None:
            # Ensure maxnorm constraints are initially satisfied
            entity_init = dense_maxnorm(entity_init, self.maxnorm)
            rel_init = dense_maxnorm(rel_init, self.maxnorm)
        self.entity_embedding_vars = tf.Variable(entity_init)
        self.rel_embedding_vars = tf.Variable(rel_init)
        # Embedding layer for each (head, rel, tail) triple being fed in as input
        head_embed = tf.nn.embedding_lookup(self.entity_embedding_vars, self.head_input)
        tail_embed = tf.nn.embedding_lookup(self.entity_embedding_vars, self.tail_input)
        rel_embed = tf.nn.embedding_lookup(self.rel_embedding_vars, self.rel_input)
        # Reshape rel_embed into square D x D matrices
        rel_embed_square = tf.reshape(rel_embed, (-1, self.embedding_size, self.embedding_size))
        # Reshape head_embed and tail_embed to be suitable for the matrix multiplication
        head_embed_row = tf.expand_dims(head_embed, 1) # embeddings as row vectors
        tail_embed_col = tf.expand_dims(tail_embed, 2) # embeddings as column vectors
        head_rel_mult = tf.batch_matmul(head_embed_row, rel_embed_square)
        # Output needs a squeeze into a 1d vector
        raw_output = tf.squeeze(tf.batch_matmul(head_rel_mult, tail_embed_col))
        self.output, self.loss = self._create_output_and_loss(raw_output)
        # Optimization
        self.train_step = self.opt.minimize(self.loss)
        if self.maxnorm is not None:
            # Post-processing to limit embedding vars to L2 ball
            rel_maxnorm = self.maxnorm * self.rel_maxnorm_mult
            unique_ent_indices = tf.unique(tf.concat(0, [self.head_input, self.tail_input]))[0]
            unique_rel_indices = tf.unique(self.rel_input)[0]
            entity_constraint = self._norm_constraint_op(self.entity_embedding_vars,
                                                         unique_ent_indices,
                                                         self.maxnorm)
            rel_constraint = self._norm_constraint_op(self.rel_embedding_vars,
                                                      unique_rel_indices,
                                                      rel_maxnorm)
            self.post_step = [entity_constraint, rel_constraint]


class TransE(BaseModel):
    ''' TransE: Translational Embeddings Model

    Score for (head i, rel k, tail j) triple is:  d(e_i + t_k, e_i)
    where e_i and e_j are D-dimensional embedding vectors for the head and
    tail entities, t_k is a another D-dimensional vector acting as a
    translation, and d() is a dissimilarity function like Euclidean distance.

    Optimization is performed uing SGD on ranking margin loss between
    contrastive training pairs. Entity embeddings are contrained to lie within
    the unit L2 ball, relationship vectors are left unconstrained.

    Args:
        embedding_size: Embedding vector length
        batch_pos_cnt: Number of positive examples to use in each mini-batch
        max_iter: Maximum number of optimization iterations to perform
        dist: Distance function used in loss:
            'euclidean': sqrt(sum((x - y)^2))
            'sqeuclidean': squared Euclidean, sum((x - y)^2)
            'manhattan': sum of absolute differences, sum(|x - y|)
        margin: Margin parameter for parwise ranking hinge loss
        opt: An optimizer object to use. If None, the default optimizer is
            tf.train.AdagradOptimizer(1.0)

    References:
    Bordes, Antoine, et al. "Translating embeddings for modeling multi-relational
    data." Advances in Neural Information Processing Systems. 2013.
    '''
    def __init__(self, embedding_size, batch_pos_cnt=100,
                 max_iter=1000, dist='euclidean',
                 margin=1.0, opt=None):
        super(TransE, self).__init__(embedding_size=embedding_size,
                                     maxnorm=1.0,
                                     batch_pos_cnt=batch_pos_cnt,
                                     max_iter=max_iter,
                                     model_type='ranking_margin',
                                     opt=opt)
        self.dist = dist
        self.margin = margin
        self.EPS = 1e-3 # for sqrt gradient when dist='euclidean'

    def _create_model(self, train_triples):
        # Count unique items to determine embedding matrix sizes
        entity_cnt = len(set(train_triples[:,0]).union(train_triples[:,2]))
        rel_cnt = len(set(train_triples[:,1]))
        init_sd = 1.0 / np.sqrt(self.embedding_size)
        # Embedding variables
        entity_var_shape = [entity_cnt, self.embedding_size]
        rel_var_shape = [rel_cnt, self.embedding_size]
        entity_init  = tf.truncated_normal(entity_var_shape, stddev=init_sd)
        rel_init = tf.truncated_normal(rel_var_shape, stddev=init_sd)
        # Ensure maxnorm constraints are initially satisfied
        entity_init = dense_maxnorm(entity_init, self.maxnorm)
        self.entity_embedding_vars = tf.Variable(entity_init)
        self.rel_embedding_vars = tf.Variable(rel_init)
        # Embedding layer for each (head, rel, tail) triple being fed in as input
        head_embed = tf.nn.embedding_lookup(self.entity_embedding_vars, self.head_input)
        tail_embed = tf.nn.embedding_lookup(self.entity_embedding_vars, self.tail_input)
        rel_embed = tf.nn.embedding_lookup(self.rel_embedding_vars, self.rel_input)
        # Relationship vector acts as a translation in entity embedding space
        diff_vec = tail_embed - (head_embed + rel_embed)
        # negative dist so higher scores are better (important for pairwise loss)
        if self.dist == 'manhattan':
            raw_output = -tf.reduce_sum(tf.abs(diff_vec), 1)
        elif self.dist == 'euclidean':
            # +eps because gradients can misbehave for small values in sqrt
            raw_output = -tf.sqrt(tf.reduce_sum(tf.square(diff_vec), 1) + self.EPS)
        elif self.dist == 'sqeuclidean':
            raw_output = -tf.reduce_sum(tf.square(diff_vec), 1)
        else:
            raise Exception('Unknown distance type')
        # Model output
        self.output, self.loss = ranking_margin_objective(raw_output, self.margin)
        # Optimization with postprocessing to limit embedding vars to L2 ball
        self.train_step = self.opt.minimize(self.loss)
        unique_ent_indices = tf.unique(tf.concat(0, [self.head_input, self.tail_input]))[0]
        self.post_step = self._norm_constraint_op(self.entity_embedding_vars,
                                                  unique_ent_indices,
                                                  self.maxnorm)
