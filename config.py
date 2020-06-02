from transformers import T5Tokenizer


BATCH_SIZE = 1
EPOCHS = 5
LEARNING_RATE = 3e-5
MAX_INPUT_LEN = 50
MAX_TARGET_LEN = 36
Tokenizer = T5Tokenizer.from_pretrained('t5-large')