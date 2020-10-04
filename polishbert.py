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