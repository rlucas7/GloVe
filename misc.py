# scipt to read in co-occurence records for subsequent
# GloVe-V processing

import argparse
import struct
from collections import deque
from math import log

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def read_cooccurrence_file(filename):
    """
    Reads the binary cooccurrence file from GloVe C implementation.
    The file format is a sequence of triples:
    (int64, int64, float64) = (word1_id, word2_id, count)
    """
    cooccurrences = []
    # 'i' = int (4 bytes), 'd' = double (8 bytes)
    # Total 24 bytes per entry
    # to inspect the *.bin file use
    # `hexdump -C cooccurrence.bin | head `
    fmt = "<iid"                    # little-endian: int, int, double
    rec_size = struct.calcsize(fmt) # should be 16
    with open(filename, 'rb') as f:
        while True:
            chunk = f.read(rec_size)
            if not chunk:
                break
            word1_id, word2_id, count = struct.unpack(fmt, chunk)
            # words are 1 indexed...
            cooccurrences.append((word1_id, word2_id, count)) # we index the map at 1 so the indices here do not change
    return cooccurrences


def load_vocabulary(vocab_path):
    """
    Creates a mapping of word IDs to word strings.
    GloVe IDs start at 1 based on the line position in vocab.txt.
    """
    id_to_word = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):  # Start indexing at 1
            word = line.split()[0]      # Format is usually 'word count'
            id_to_word[i] = word
    return id_to_word


def calculate_sigma_i_sq(vectors_path, ith_word_id, paired_word_ids, D=50, x_max=100, alpha=0.75):
    """
    Calculates the scalar estimate for the variance using eqn 7
    from the GloVe-V paper. This is an approximation which ignores
    the covariance terms for computational efficiency. Doing so
    obviates the inversion of a dense D x D matrix for each word
    in the corpus.
    """
    # assuming dimension 50 for now but this should be input
    # ---- # get the ith data, b_i, w_i in the first pass.
    with open(vectors_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):  # Start indexing at 1
            if i == ith_word_id:
                line_items = line.rstrip('\n').split(' ')
                # print(i, len(line_items), line_items)
                word_i = line_items[0]
                w_i = [float(v) for v in line_items[1:D+1]]
                b_i = float(line_items[D+1:D+2][0])
                # omit v_i, c_i
    # ----
    # Now on the second pass, go for the pairs and accumulate
    sigma_i_sq = 0.0
    id_xij_pairs = deque(paired_word_ids.items())
    _id, x_ij = id_xij_pairs.popleft()
    with open(vectors_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):  # Start indexing at 1
            if i == _id:
                line_items = line.rstrip('\n').split(' ')
                # print(i, len(line_items), line_items)
                wordj = line_items[0]
                # for the fixed word (w_i, b_i), will be for the ith word
                # here for completeness we parse the w_j, b_j and omit
                # TODO: consider a binary file write for less memory
                w_j = [float(v) for v in line_items[1:D+1]]
                b_j = float(line_items[D+1:D+2][0])
                v_j = [float(v) for v in line_items[D+2:2*D+2]]
                c_j = float(line_items[2*D+2])
                # calculate log(X_ij) - b_i - c_j - w_i^Tv_j
                val = log(x_ij) - b_i - c_j
                for j in range(D):
                    val -= w_i[j] * v_j[j]
                # If non-defaults used in training then,
                # the x_max, alpha = 100, 0.75 values should
                # be updated to reflect the non-default values used
                f_ij = pow(x_ij / x_max, alpha)  if x_ij <= x_max else 1
                sigma_i_sq += f_ij * val * val
                # update
                if id_xij_pairs:
                    _id, x_ij = id_xij_pairs.popleft()
                else:
                    break
    if id_xij_pairs:
        # we should never have left over pairs that are unprocessed
        print(len(id_xij_pairs))
    # divide by |K|-D in equation (7) from GloVe-V
    sigma_i_sq /= len(paired_word_ids) - D
    return sigma_i_sq


# Evaluates Equation 6 the dense variance covariance matrix.
# Note this is on demand and for a single word so it isn't onerous computationally but
# would become so if we did this for all Vocabulary words.


def calc_dense_var_covar(vectors_path, ith_word_id, paired_word_ids, D=50, x_max=100, alpha=0.75):
    """ Calculate the dense matrix for eqn 6 of the GloVe-V paper. This function ignores the
    $ sigma_i^2$ scalar term because that is calculated with `calculate_sigma_i_sq`.
    In other words, we calculate $ sum_k f(Xij)*v_jv_j^T$ and return this D x D matrix.
    """
    var_covar = np.zeros((D, D), dtype=float)
    id_xij_pairs = deque(paired_word_ids.items())
    _id, x_ij = id_xij_pairs.popleft()
    with open(vectors_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):  # Start indexing at 1
            if i == _id:
                line_items = line.rstrip('\n').split(' ')
                # print(i, len(line_items), line_items)
                wordj = line_items[0]
                # for the fixed word (w_i, b_i), will be for the ith word
                # here for completeness we parse the w_j, b_j and omit
                # TODO: consider a binary file write for less memory
                v_j = np.array([float(v) for v in line_items[D+2:2*D+2]])
                f_ij = pow(x_ij / x_max, alpha)  if x_ij <= x_max else 1
                var_covar += f_ij * np.outer(v_j, v_j)
                # update
                if id_xij_pairs:
                    _id, x_ij = id_xij_pairs.popleft()
                else:
                    break
    return var_covar


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("-vo", "--vocab", type=str, default="vocab.txt",  help="The vocab file from GloVe.")
parser.add_argument("-c", "--cooccur", type=str, default="cooccurrences.bin", help="The cooccurrences file from GloVe.")
parser.add_argument("-ve", "--vectors", type=str, default="vectors.txt",  help="The vectors file from GloVe.")
parser.add_argument("-p", "--plot", action="store_true", help="When the plot flag is set the script plots a heatmap of var-covar matrix for the indicated word.")
parser.add_argument("--var-covar-npy", type=str, default=None, help="Optional path to save the variance-covariance matrix as a NumPy .npy file.")
parser.add_argument("--plot-file", type=str, default=None, help="Optional path to save the variance-covariance heatmap figure.")
parser.add_argument("-d", "--dim", type=int, default=50,  help="""The dimensionality of the vectors file from GloVe.
    Note that this dimension is the size of the embeddings and does not correspond to the length of the vector file in the
    glove output in the usual output format. The reason is that we use the -method 0 flag when calling GloVe executable.
    This causes glove vector to be 50 + 1 + 50 + 1 = 102 floats rather than 50 using the default `-method` value.
    We do this because both the biases (aka offsets, +1s in the tally above) are necessary as are the context vectors.
    The context vectors and the word vectors are both output with the offsets using `-method 0` flag. If you have not done
    so you will get erroneous output.
    """)

## add mutex group so user may specify either word id-useful in scripting/looping- or word as text
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-wi", "--word-id", type=int, default=None,
    help="The id of the word from the vocab file to calculate Variance-Cov matrix")
group.add_argument("-w", "--word", type=str, default=None,
    help="The word from the vocab file to calculate Variance-Cov matrix")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    # Example usage:
    id2word = load_vocabulary(args.vocab)
    word2id = {word: idx for idx, word in id2word.items()}
    cooccur = read_cooccurrence_file(args.cooccur)

    # Resolve word selection from either id or token.
    if args.word_id is not None:
        if args.word_id not in id2word:
            raise RuntimeError(f"Word id {args.word_id} does not exist in the corpus vocabulary.")
        v = args.word_id
    else:
        requested_word = args.word
        if requested_word in word2id:
            v = word2id[requested_word]
        else:
            fallback_id = None
            fallback_token = None
            for token in ("<unk>", "``"):
                if token in word2id:
                    fallback_id = word2id[token]
                    fallback_token = token
                    break
            if fallback_id is None:
                raise RuntimeError(
                    f"Word '{requested_word}' does not exist in the corpus vocabulary, and no fallback token "
                    f"('<unk>' or '``') is available."
                )
            print(f"Word '{requested_word}' does not exist in the corpus; using fallback token '{fallback_token}'.")
            v = fallback_id

    # some static code to fill in a word id map
    word_idx_c = dict()
    for i in range(len(cooccur)):
        w1, w2, f = cooccur[i]
        word1 = id2word[w1]
        word2 = id2word[w2]
        if w1 == v or w2 == v:
            # the print is optional...
            # print(word1, word2, f)
            if w1 != v:
                word_idx_c[w1] = f
            elif w2 != v:
                word_idx_c[w2] = f

    if not word_idx_c:
        raise RuntimeError(f"No cooccurrence pairs found for word '{id2word[v]}' (id={v}).")

    # for word 9999 of text8 data, this is 'antibiotics' which is still somewhat common.
    # figure 2 of the GloVe-V paper indicates an inverse relationship between word frequency
    # and word variance.
    sigma_i_sq = calculate_sigma_i_sq(vectors_path=args.vectors, ith_word_id=v, paired_word_ids=word_idx_c, D=args.dim)
    v_cov = calc_dense_var_covar(vectors_path=args.vectors, ith_word_id=v, paired_word_ids=word_idx_c, D=args.dim)
    var_covar = sigma_i_sq*np.linalg.inv(v_cov)
    print(f"Variance-Covariance matrix of: {id2word[v]}")
    print(var_covar)
    if args.var_covar_npy:
        np.save(args.var_covar_npy, var_covar)
        print(f"Wrote variance-covariance matrix to: {args.var_covar_npy}")

    if args.plot or args.plot_file:
        try:
            plt.figure()
            sns.heatmap(var_covar)
            plt.title(f"Var-Covariance of GloVe-V Embedding\nDimensions for word: {id2word[v]}")
            if args.plot_file:
                plt.savefig(args.plot_file, bbox_inches="tight")
                print(f"Wrote variance-covariance plot to: {args.plot_file}")
            if args.plot:
                plt.show()
        finally:
            plt.close()

    # NOTE: this is some wip code for nows with various word cooccur frequencies
    # the rest of this code is highly specific to text8 data and will need some
    # care to make it more applicable to other corpus than text8.

    # some code to investigate the impact of cooccur frequencies on the scalar variance
    # from GloVe-V

    #so how many words cooccur with 21 -> "four"? -> 47115 still large
    #so how many words cooccur with 40  -> "were"? -> 35993 still large
    #so how many words cooccur with 107  -> 'under'? -> 20971 still large
    #so how many words cooccur with 203  -> 'order'? -> 15498 still large
    #so how many words cooccur with 507  -> 'study'? -> 9384 still large
    #so how many words cooccur with 1003  -> 'existence'? -> 6317 still large
    #so how many words cooccur with 5003  -> 'robin'? -> 2168 still large
    #so how many words cooccur with 7005  -> 'slot'? ->  1355 still large

    #so how many words cooccur with 9007  -> 'huxley'? ->  1043 still large
    #so how many words cooccur with 9999  -> 'antibiotics'? -> 851 this will do for now
    #so how many words cooccur with 10607 -> ''lockheed' ? -> 741 this will do for now
    #so how many words cooccur with 12001 -> 'expectation' ? -> 801 this will do for now
    #so how many words cooccur with 16021 -> 'importation' ? -> 560 this will do for now
    #so how many words cooccur with 18007 -> 'sentient' ? -> 500 this will do for now
    #so how many words cooccur with 19007 ->  'charities' ? -> 473  this will do for now
    #so how many words cooccur with 20001 -> 'ui' ? -> 388 this will do for now
    # a pretty good spread for these 6 words...
