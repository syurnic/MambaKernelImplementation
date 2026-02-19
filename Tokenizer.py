from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. Load the dataset
ds = load_dataset("roneneldan/TinyStories", split="train")

# 2. Define a generator to feed text to the trainer efficiently
def batch_iterator():
    for i in range(0, len(ds), 1000):
        yield ds[i : i + 1000]["text"]

# 3. Initialize the BPE Tokenizer
# We use ByteLevel BPE (like GPT-2) to handle any character gracefully
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 4. Configure the Trainer
# 8,000 to 16,000 is a great vocab size for TinyStories
trainer = trainers.BpeTrainer(
    vocab_size=5000,
    special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
)

# 5. Train!
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

# 6. Save the files for your C++ project
tokenizer.save("mamba_vocab.json")
print("Vocabulary generated and saved!")