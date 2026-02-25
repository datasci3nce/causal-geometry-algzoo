from .architectures import DistRNN, DistTransformer, ScalarRNN
from .loading import load_dist_rnn, load_dist_transformer, load_scalar_rnn

ZOO_BASEPATH = "gs://arc-ml-public/alg/zoo"


def check_prop_in(key, value, values):
    if value not in values:
        raise ValueError(
            f"No zoo model with {key} {value}, "
            f"please choose one of: {', '.join(map(str, values))}"
        )


def check_prop_match(key, value, model):
    assert getattr(model, key) == value, (
        f"Model {key} {getattr(model, key)} " f"did not match requested {key} {value}"
    )


def zoo_2nd_argmax(hidden_size, seq_len, *, n_seqs=None, seed=None) -> DistRNN:
    """
    Load a 2nd_argmax RNN from AlgZoo.

    Arguments:
    - hidden_size: number of neurons in each hidden layer
    - seq_len: length of input sequence
    - n_seqs: number of training sequences, for a partially-trained model.
        defaults to fully-trained model
    - seed: manual torch seed used for training, from 0 to 4.
        defaults to best-of-5 seed based on accuracy
    """
    check_prop_in("hidden_size", hidden_size, [2, 3, 4, 5, 6, 8, 16, 32])
    check_prop_in("seq_len", seq_len, [2, 3, 4, 5, 6, 8, 10])
    if n_seqs is None:
        n_seqs = 2**30
    check_prop_in("n_seqs", n_seqs, [0] + [2**i for i in range(20, 31)])
    if seed is None:
        seed = best_of_5_seed_2nd_argmax(hidden_size=hidden_size, seq_len=seq_len)
    check_prop_in("seed", seed, list(range(5)))
    model = load_dist_rnn(
        f"{ZOO_BASEPATH}/2nd_argmax_{hidden_size}_{seq_len}_{n_seqs}_{seed}.pth",
        bias=False,
    )
    check_prop_match("hidden_size", hidden_size, model)
    check_prop_match("seq_len", seq_len, model)
    return model


def zoo_argmedian(hidden_size, seq_len, *, n_seqs=None, seed=None) -> DistRNN:
    """
    Load an argmedian RNN from AlgZoo.

    Arguments:
    - hidden_size: number of neurons in each hidden layer
    - seq_len: length of input sequence
    - n_seqs: number of training sequences, for a partially-trained model.
        defaults to fully-trained model
    - seed: manual torch seed used for training, from 0 to 4.
        defaults to best-of-5 seed based on accuracy
    """
    check_prop_in("hidden_size", hidden_size, [2, 4, 8, 16, 32])
    check_prop_in("seq_len", seq_len, [3, 5, 7, 11])
    if n_seqs is None:
        n_seqs = 2**30
    check_prop_in("n_seqs", n_seqs, [0] + [2**i for i in range(20, 31)])
    if seed is None:
        seed = best_of_5_seed_argmedian(hidden_size=hidden_size, seq_len=seq_len)
    check_prop_in("seed", seed, list(range(5)))
    model = load_dist_rnn(
        f"{ZOO_BASEPATH}/argmedian_{hidden_size}_{seq_len}_{n_seqs}_{seed}.pth",
        bias=False,
    )
    check_prop_match("hidden_size", hidden_size, model)
    check_prop_match("seq_len", seq_len, model)
    return model


def zoo_median(hidden_size, seq_len, *, n_seqs=None, seed=None) -> ScalarRNN:
    """
    Load a median RNN from AlgZoo.

    Arguments:
    - hidden_size: number of neurons in each hidden layer
    - seq_len: length of input sequence
    - n_seqs: number of training sequences, for a partially-trained model.
        defaults to fully-trained model
    - seed: manual torch seed used for training, from 0 to 4.
        defaults to best-of-5 seed based on mean squared error
    """
    check_prop_in("hidden_size", hidden_size, [2, 4, 8, 16, 32])
    check_prop_in("seq_len", seq_len, [2, 3, 5, 10])
    if n_seqs is None:
        n_seqs = 2**30
    check_prop_in("n_seqs", n_seqs, [0] + [2**i for i in range(20, 31)])
    if seed is None:
        seed = best_of_5_seed_median(hidden_size=hidden_size, seq_len=seq_len)
    check_prop_in("seed", seed, list(range(5)))
    model = load_scalar_rnn(
        f"{ZOO_BASEPATH}/median_{hidden_size}_{seq_len}_{n_seqs}_{seed}.pth",
        seq_len=seq_len,
        bias=False,
    )
    check_prop_match("hidden_size", hidden_size, model)
    return model


def zoo_longest_cycle(
    hidden_size, seq_len, *, n_seqs=None, seed=None
) -> DistTransformer:
    """
    Load a longest cycle transformer from AlgZoo.

    Arguments:
    - hidden_size: number of neurons in each hidden layer
    - seq_len: length of input sequence
    - n_seqs: number of training sequences, for a partially-trained model.
        defaults to fully-trained model
    - seed: manual torch seed used for training, from 0 to 4.
        defaults to best-of-5 seed based on accuracy
    """
    check_prop_in("hidden_size", hidden_size, [2, 3, 4, 6, 8])
    check_prop_in("seq_len", seq_len, [3, 4, 5, 6])
    if n_seqs is None:
        n_seqs = 2**30
    check_prop_in("n_seqs", n_seqs, [0] + [2**i for i in range(20, 31)])
    if seed is None:
        seed = best_of_5_seed_longest_cycle(hidden_size=hidden_size, seq_len=seq_len)
    check_prop_in("seed", seed, list(range(5)))
    model = load_dist_transformer(
        f"{ZOO_BASEPATH}/longest_cycle_{hidden_size}_{seq_len}_{n_seqs}_{seed}.pth",
        n_heads=1,
        bias=False,
    )
    check_prop_match("hidden_size", hidden_size, model)
    check_prop_match("seq_len", seq_len, model)
    return model


def best_of_5_seed_2nd_argmax(hidden_size, seq_len):
    return {
        (2, 2): 2,
        (2, 3): 0,
        (2, 4): 2,
        (2, 5): 3,
        (2, 6): 2,
        (2, 8): 1,
        (2, 10): 4,
        (3, 2): 3,
        (3, 3): 1,
        (3, 4): 4,
        (3, 5): 0,
        (3, 6): 0,
        (3, 8): 1,
        (3, 10): 1,
        (4, 2): 3,
        (4, 3): 0,
        (4, 4): 1,
        (4, 5): 2,
        (4, 6): 0,
        (4, 8): 1,
        (4, 10): 2,
        (5, 2): 3,
        (5, 3): 1,
        (5, 4): 4,
        (5, 5): 3,
        (5, 6): 4,
        (5, 8): 4,
        (5, 10): 4,
        (6, 2): 2,
        (6, 3): 3,
        (6, 4): 0,
        (6, 5): 1,
        (6, 6): 1,
        (6, 8): 0,
        (6, 10): 2,
        (8, 2): 3,
        (8, 3): 1,
        (8, 4): 1,
        (8, 5): 3,
        (8, 6): 4,
        (8, 8): 0,
        (8, 10): 4,
        (16, 2): 3,
        (16, 3): 1,
        (16, 4): 3,
        (16, 5): 2,
        (16, 6): 3,
        (16, 8): 1,
        (16, 10): 2,
        (32, 2): 3,
        (32, 3): 4,
        (32, 4): 2,
        (32, 5): 3,
        (32, 6): 1,
        (32, 8): 4,
        (32, 10): 0,
    }[(hidden_size, seq_len)]


def best_of_5_seed_argmedian(hidden_size, seq_len):
    return {
        (2, 3): 0,
        (2, 5): 0,
        (2, 7): 0,
        (2, 11): 3,
        (4, 3): 0,
        (4, 5): 2,
        (4, 7): 1,
        (4, 11): 1,
        (8, 3): 1,
        (8, 5): 2,
        (8, 7): 0,
        (8, 11): 4,
        (16, 3): 1,
        (16, 5): 0,
        (16, 7): 4,
        (16, 11): 4,
        (32, 3): 4,
        (32, 5): 3,
        (32, 7): 3,
        (32, 11): 4,
    }[(hidden_size, seq_len)]


def best_of_5_seed_median(hidden_size, seq_len):
    return {
        (2, 2): 2,
        (2, 3): 0,
        (2, 5): 0,
        (2, 10): 0,
        (4, 2): 4,
        (4, 3): 0,
        (4, 5): 4,
        (4, 10): 1,
        (8, 2): 3,
        (8, 3): 3,
        (8, 5): 1,
        (8, 10): 3,
        (16, 2): 1,
        (16, 3): 1,
        (16, 5): 0,
        (16, 10): 2,
        (32, 2): 1,
        (32, 3): 1,
        (32, 5): 1,
        (32, 10): 0,
    }[(hidden_size, seq_len)]


def best_of_5_seed_longest_cycle(hidden_size, seq_len):
    return {
        (2, 3): 0,
        (2, 4): 2,
        (2, 5): 3,
        (2, 6): 2,
        (3, 3): 0,
        (3, 4): 4,
        (3, 5): 1,
        (3, 6): 0,
        (4, 3): 2,
        (4, 4): 1,
        (4, 5): 0,
        (4, 6): 0,
        (6, 3): 0,
        (6, 4): 0,
        (6, 5): 2,
        (6, 6): 3,
        (8, 3): 0,
        (8, 4): 3,
        (8, 5): 3,
        (8, 6): 3,
    }[(hidden_size, seq_len)]
