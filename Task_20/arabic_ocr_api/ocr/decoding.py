"""
CTC decoding for inference. Mirrors decode_batch_predictions_beam() from the
training notebook. Beam search is used here (rather than greedy) because a
single request's extra latency matters far less than getting the text
right.
"""

import numpy as np
import tensorflow as tf


def decode_predictions_beam(preds: np.ndarray, idx_to_char: dict, beam_width: int = 10):
    """
    preds: model output, shape (batch, time_steps, vocab_size + 1)
    idx_to_char: {index: character} built from vocab.json, where index 0 is
                 the StringLookup OOV/pad token and must be dropped, and -1
                 marks CTC blanks and must also be dropped.
    """
    input_len = np.ones(preds.shape[0]) * preds.shape[1]

    results = tf.keras.backend.ctc_decode(
        preds, input_length=input_len, greedy=False, beam_width=beam_width
    )[0][0].numpy()

    texts = []
    for seq in results:
        chars = [idx_to_char.get(int(idx), "") for idx in seq if idx not in (-1, 0)]
        texts.append("".join(chars))
    return texts
