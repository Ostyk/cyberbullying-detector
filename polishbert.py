import torch
from tokenizers import SentencePieceBPETokenizer
from tokenizers.processors import RobertaProcessing
from transformers import RobertaModel, AutoModel
import numpy as np

def load_embeddings(model_dir = "roberta_base_transformers"):
    """
    Function that loads a pre-trained polish BErt model used for embedding extraction
    :param model_dir: full path to the directory with model files
    :return: model, tokenizer
    """

    tokenizer = SentencePieceBPETokenizer(f"{model_dir}/vocab.json", f"{model_dir}/merges.txt")
    getattr(tokenizer, "_tokenizer").post_processor = RobertaProcessing(sep=("</s>", 2), cls=("<s>", 0))
    model: RobertaModel = AutoModel.from_pretrained(model_dir)
    return model, tokenizer


def feed_batch(x, batch_size):
    """
    Function that feeds batches of embeddings
    :param x: all indicies
    :param batch_size:
    :return: depends on the current index
    """
    i, batch = 0, 0
    for batch, i in enumerate(range(0, len(x) - batch_size, batch_size), 1):
        x_batch = x[i : i + batch_size]
        yield x_batch, batch
    if i + batch_size < len(x):
        yield x[i + batch_size :], batch + 1
    if batch == 0:
        yield x, 1
        
def extract_in_batches(indicies, embeddings, batch_size=30):
    """
    Function that extracts embeddings in batches due to OOM issues
    :param indicies: all indicies
    :param embeddings: embeddings
    :param batch_size:
    :return: new embeddings
    """
    new = torch.zeros((*indicies.shape, 768))
    previous = 0
    with torch.no_grad():
        for batch, num in feed_batch(indicies, batch_size):
            new[previous*batch_size:num*batch_size] = embeddings(batch)[0]
            previous = num
    return new


def tokenize_text(df, max_seq, tokenizer):
    return [
        tokenizer.encode(text, add_special_tokens=True).ids[:max_seq] for text in df.values]
def pad_text(tokenized_text, max_seq):
    return np.array([el + [0] * (max_seq - len(el)) for el in tokenized_text])
def tokenize_and_pad_text(df, max_seq, tokenizer):
    tokenized_text = tokenize_text(df, max_seq, tokenizer)
    padded_text = pad_text(tokenized_text, max_seq)
    return torch.tensor(padded_text)
def targets_to_tensor(df, target_columns):
    return torch.tensor(df[target_columns].values, dtype=torch.float32)


