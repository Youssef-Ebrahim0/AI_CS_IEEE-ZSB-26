"""
Loads the trained CRNN model, vocabulary, and config exactly once, at
server startup (see app.py's lifespan handler). Loading a Keras model per
request would be far too slow and would leak memory over time.
"""

import logging

import tensorflow as tf

from . import config as cfg

logger = logging.getLogger("ocr.model_loader")


class OCRModel:
    """Holds the loaded Keras model + vocab + config as a single object,
    populated once and shared across all requests via app.state / import."""

    def __init__(self):
        self.model = None
        self.vocab = None
        self.config = None
        self.idx_to_char = None

    def load(self) -> "OCRModel":
        logger.info("Loading OCR config...")
        self.config = cfg.load_config()

        logger.info("Loading vocabulary...")
        self.vocab = cfg.load_vocab()
        # vocab[0] is the StringLookup OOV token; real characters start at
        # index 1, matching char_to_num used during training.
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}

        logger.info("Loading TensorFlow model from %s ...", cfg.MODEL_PATH)
        # crnn_model (the inference model) contains no custom layers --
        # CTCLayer only lives inside the training wrapper -- so no
        # custom_objects registration is needed here.
        self.model = tf.keras.models.load_model(cfg.MODEL_PATH)

        # Warm up the model with one dummy forward pass so the first real
        # request isn't slowed down by lazy graph tracing / XLA warmup.
        img_w = self.config["IMG_WIDTH"]
        img_h = self.config["IMG_HEIGHT"]
        dummy = tf.zeros((1, img_w, img_h, 1), dtype=tf.float32)
        self.model.predict(dummy, verbose=0)

        logger.info(
            "OCR model ready. vocab_size=%d, ctc_time_steps=%d",
            self.config["VOCAB_SIZE"],
            self.config["CTC_TIME_STEPS"],
        )
        return self


# Singleton populated once at server startup and shared across all requests.
ocr_model = OCRModel()
