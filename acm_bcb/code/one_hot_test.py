import numpy as np
import tensorflow as tf

class YourClass:
    def __init__(self):
        self.MAX_LEN = 100  # Set your max length
        self._n_chars = 50  # Set the number of characters in your vocabulary
        self._prod_map = {}  # Initialize your production map
        self._tokenize = lambda x: x  # Dummy tokenize function
        self._parser = self.DummyParser()  # Dummy parser class

    class DummyParser:
        def parse(self, tokens):
            # Dummy parse function that returns an iterable with one element
            yield []  # Replace with actual parsing logic

    @tf.function
    def onehot(self, smiles):
        """ Encode a list of smiles strings into the latent space """
        one_hot = tf.zeros((self.MAX_LEN, self._n_chars), dtype=tf.float32)
        
        # Convert SMILES to tokens
        tokens = tf.map_fn(self._tokenize, smiles, dtype=tf.string)
        
        def encode_one_smiles(smiles):
            try:
                tp = next(iter(self._parser.parse(smiles)))
            except Exception as e:
                tf.print("Parse tree error:", e)
                return one_hot

            productions_seq = tp  # Assuming productions_seq is already a tensor
            idx = tf.constant([self._prod_map.get(prod, 0) for prod in productions_seq], dtype=tf.int32)
            num_productions = tf.shape(idx)[0]

            def true_fn():
                return tf.tensor_scatter_nd_update(one_hot, tf.range(self.MAX_LEN)[:, None], tf.ones((self.MAX_LEN,), dtype=tf.float32))

            def false_fn():
                updated_one_hot = tf.tensor_scatter_nd_update(one_hot, tf.range(num_productions)[:, None], tf.ones((num_productions,), dtype=tf.float32))
                return tf.tensor_scatter_nd_update(updated_one_hot, tf.range(num_productions, self.MAX_LEN)[:, None], tf.ones((self.MAX_LEN - num_productions,), dtype=tf.float32))

            one_hot = tf.cond(num_productions > self.MAX_LEN, true_fn, false_fn)
            return one_hot

        one_hot = tf.map_fn(encode_one_smiles, tokens, dtype=tf.float32)
        return one_hot

# Usage
your_class_instance = YourClass()
smiles_list = tf.constant(["CCO", "NCC"], dtype=tf.string)
one_hot_result = your_class_instance.onehot(smiles_list)
print(one_hot_result)
