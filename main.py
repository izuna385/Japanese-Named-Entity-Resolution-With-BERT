from dataset_reader import NamedEntityResolutionReader
from parameters import Params
import pdb
from utils import build_vocab, build_data_loaders, build_trainer, emb_returner
from encoder import Pooler_for_mention
from model import ResolutionLabelClassifier

if __name__ == '__main__':
    params = Params()
    config = params.opts
    dsr = NamedEntityResolutionReader(config=config)

    # Loading Datasets
    train, dev, test = dsr._read('train'), dsr._read('dev'), dsr._read('test')
    train_and_dev = train + dev
    vocab = build_vocab(train_and_dev)

    train_loader, dev_loader = build_data_loaders(config, train, dev)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    _, __, embedder = emb_returner(config=config)
    mention_encoder = Pooler_for_mention(config, embedder)

    model = ResolutionLabelClassifier(config, mention_encoder, vocab)

    trainer = build_trainer(config, model, train_loader, dev_loader)
    trainer.train()