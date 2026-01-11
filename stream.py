#!/usr/bin/env python3
"""
Streaming writer for HF Arrow datasets -> stdout (one example per line).
Handles BrokenPipe / SIGPIPE so it exits cleanly when consumer closes.
The first example shows the use of sed to filter specific tokens in the
`std{in,out}` stream.
Usage examples:

```
  python stream.py | sed -e 's/<unk>/<raw_unk>/g' | ./build/vocab_count -min-count 50 -verbose 2 > vocab.txt
  python stream.py | ./build/cooccur -memory 1024 -vocab-file vocab.txt -window-size 10 > cooccurrence.bin
```

Then to do the GloVe training you can do:

```
  OMP_NUM_THREADS=4 ./build/glove -save-file vectors \
                                          -threads 4 \
                  -input-file cooccurrences.shuf.bin \
                               -vocab-file vocab.txt \
                                     -vector-size 50 \
                                            -iter 50 \
                                          -x-max 100 \
                                           -binary 2 \
                                          -verbose 2 \
                                           -eta 0.05
```
Notice that here we pass in 4 openMP threads but this isn't necessary.
"""
import sys
import signal
import argparse
from itertools import chain
from datasets import load_from_disk, Dataset

# Let the default SIGPIPE behavior terminate the process quietly instead of raising exceptions.
# This avoids printing a BrokenPipe traceback when the consumer exits.
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

import os

# Load the dataset: TODO: clean this part up so it works to load more generally than
# the string literals here...
local_path = "~/.cache/huggingface/datasets/wikitext/wikitext-103-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3"
full_path = os.path.expanduser(local_path)

def stream_wikitext(lowercase: bool = False, show_progress: bool = True):
    #ds_train = load_from_disk(local_path)
    ds_train_1 = Dataset.from_file(f"{full_path}/wikitext-train-00000-of-00002.arrow")
    ds_train_2 = Dataset.from_file(f"{full_path}/wikitext-train-00001-of-00002.arrow")
    ds_valid = Dataset.from_file(f"{full_path}/wikitext-validation.arrow")
    ds_test  = Dataset.from_file(f"{full_path}/wikitext-test.arrow")
    out = sys.stdout.buffer
    written = 0
    try:
        #for ex in chain(ds_train, ds_valid, ds_test):
        for ex in chain(ds_train_1, ds_train_2, ds_valid, ds_test):
            text = ex.get("text", "")
            if text is None:
                continue
            if lowercase:
                text = text.lower()
            if not text.endswith("\n"):
                text = text + "\n"
            out.write(text.encode("utf-8"))
            written += 1
            if show_progress and (written % 100000 == 0):
                # print progress to stderr so it doesn't interfere with stdout->pipe
                print(f"written {written} lines", file=sys.stderr)
    except BrokenPipeError:
        # downstream closed pipe: exit quietly
        try:
            out.close()
        except Exception:
            pass
        sys.exit(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lower", action="store_true", help="lowercase text")
    ap.add_argument("--no-progress", action="store_true", help="disable stderr progress prints")
    args = ap.parse_args()
    stream_wikitext(lowercase=args.lower, show_progress=not args.no_progress)

if __name__ == "__main__":
    main()
