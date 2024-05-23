# import tensorflow as tf
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Lambda, RepeatVector, GRU, TimeDistributed
# from tensorflow.keras import Model
# import numpy as np
import numpy as np
import molecule_vae

# Assuming the zinc_grammar module provides these variables/functions
# charset = G.grammar_rules
weights_file =  '../data/vae.hdf5'

grammar_model = molecule_vae.ZincGrammarModel(weights_file)

# Test the model
# Generate a random input to test encoding and decoding process
test_input = np.random.rand(1, 277, 76)  # Example input

# Encode the input
# encoded = molecule_vae.encoder.predict(test_input)

# Decode the encoded input
# decoded = molecule_vae.decoder.predict(encoded)

# print("Encoded representation: ", encoded)
# print("Decoded output: ", decoded)
