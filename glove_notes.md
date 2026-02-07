Profiling commands on mac M4

# profile the vocab build via xcrun...
```sh
xcrun xctrace record --template "Time Profiler" --output vocab_trace.trace --launch -- /bin/zsh -c './build/vocab_count -min-count 50 < text8 > vocab.txt'
```

stdout shows:
```sh
Starting recording with the Time Profiler template. Launching process: zsh.
Ctrl-C to stop the recording
Target app exited, ending recording...
Recording completed. Saving output file...
Output file saved as: vocab_trace.trace
```


# from inspecting, one hotspot seems to be the `fgetc` call in `get_word`, so add the min patch

```sh
rlucas7@Mac glove % git diff --staged
diff --git a/src/vocab_count.c b/src/vocab_count.c
index 6e282ab..0116c52 100644
--- a/src/vocab_count.c
+++ b/src/vocab_count.c
@@ -147,6 +147,10 @@ int get_counts(void) {
 }

 int main(int argc, char **argv) {
+    /* Increase stdin buffering to reduce per-char stdio locks (fgetc/flockfile cost) */
+    setvbuf(stdin, NULL, _IOFBF, 1<<20); /* 1 MiB buffer; must be before any reads from stdin */
+
+    /* existing initialization code follows... */
     if (argc == 2 &&
         (!scmp(argv[1], "-h") || !scmp(argv[1], "-help") || !scmp(argv[1], "--help"))) {
         printf("Simple tool to extract unigram counts\n");
```
to allocate 1MiB for the stdin buffer instead of the system default.
Hopefully this speeds up somewhat the 1.15 seconds taken for `fgetc`.

Now rerun...

```bash
make clean
make
```

that rebuilds everything, move the vocab trace file to a new value to avoid overwrite

```sh
mv vocab_trace.trace init_vocab_trace.trace
```
and rerun the profiler...

now it seems like the 1.15 seconds is not shaved off the tiny `text8` input by using the slighltly larger buffering value, the new value is 1.42 seconds but it isn't clear if this change speeds up the wall clock time or not.


Going to do some testing

```sh
/usr/bin/time -l -h -p ./build/vocab_count -min-count 0 < text8 > /dev/null
/usr/bin/time ./build/vocab_count -min-count 50 < text8 > /dev/null
```

To get:
```sh
BUILDING VOCABULARY
Processed 17005207 tokens.
Counted 253854 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 18497.

real 1.70
user 1.68
sys 0.01
            27312128  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
                1861  page reclaims
                   1  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                   0  voluntary context switches
                  10  involuntary context switches
         37333374565  instructions retired
          7184719178  cycles elapsed
            27148648  peak memory footprint
rlucas7@Mac glove %


BUILDING VOCABULARY
Processed 17005207 tokens.
Counted 253854 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 18497.

real 1.69
user 1.68
sys 0.01
            27148288  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
                1851  page reclaims
                   1  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                   0  voluntary context switches
                   6  involuntary context switches
         37333522002  instructions retired
          7123601562  cycles elapsed
            26968400  peak memory footprint

BUILDING VOCABULARY
Processed 17005207 tokens.
Counted 253854 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 18497.

real 1.73
user 1.71
sys 0.01
            27279360  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
                1859  page reclaims
                   1  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                   0  voluntary context switches
                   7  involuntary context switches
         37333372860  instructions retired
          7250590084  cycles elapsed
            27099472  peak memory footprint

BUILDING VOCABULARY
Processed 17005207 tokens.
Counted 253854 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 18497.

real 1.70
user 1.69
sys 0.01
            27213824  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
                1855  page reclaims
                   1  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                   0  voluntary context switches
                   8  involuntary context switches
         37335707307  instructions retired
          7205762559  cycles elapsed
            27050344  peak memory footprint
```

then doing `make clean && make` again after commenting out the `setvbuf` line change. I redo.

Numbers without the setvbuf line...

```sh
BUILDING VOCABULARY
Processed 17005207 tokens.
Counted 253854 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 18497.

real 1.68
user 1.66
sys 0.02
            26230784  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
                1795  page reclaims
                   1  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                   0  voluntary context switches
                  13  involuntary context switches
         37447106910  instructions retired
          7074932421  cycles elapsed
            26050896  peak memory footprint

BUILDING VOCABULARY
Processed 17005207 tokens.
Counted 253854 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 18497.

real 1.73
user 1.71
sys 0.02
            26148864  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
                1790  page reclaims
                   1  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                   0  voluntary context switches
                  13  involuntary context switches
         37449229232  instructions retired
          7362099694  cycles elapsed
            25968976  peak memory footprint

BUILDING VOCABULARY
Processed 17005207 tokens.
Counted 253854 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 18497.

real 1.74
user 1.71
sys 0.02
            26247168  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
                1796  page reclaims
                   1  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                   0  voluntary context switches
                  12  involuntary context switches
         37447403585  instructions retired
          7324836770  cycles elapsed
            26067280  peak memory footprint

BUILDING VOCABULARY
Processed 17005207 tokens.
Counted 253854 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 18497.

real 1.73
user 1.70
sys 0.02
            26148864  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
                1790  page reclaims
                   1  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                   0  voluntary context switches
                  11  involuntary context switches
         37450560327  instructions retired
          7346594632  cycles elapsed
            25985384  peak memory footprint

BUILDING VOCABULARY
Processed 17005207 tokens.
Counted 253854 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 18497.

real 1.71
user 1.68
sys 0.02
            26148864  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
                1790  page reclaims
                   1  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                   0  voluntary context switches
                   9  involuntary context switches
         37449617564  instructions retired
          7197728504  cycles elapsed
            25968976  peak memory footprint

```

Instead I'm going to try this one:

```sh
for i in 1 2 3 4 5;
do
  /usr/bin/time -l -h -p python3 stream.py | sed -e 's/<unk>/<raw_unk>/g' | ./build/vocab_count -min-count 50 -verbose 2 > vocab.txt
done;
```

This uses the wikitext-103-v1 as a second file example...

Without the buffering optimization...
```sh
lucas7@Mac glove % for i in 1 2 3 4 5;
do
  /usr/bin/time -l -h -p python3 stream.py | sed -e 's/<unk>/<raw_unk>/g' | ./build/vocab_count -min-count 50 -verbose 2 > vocab.txt
done;

BUILDING VOCABULARY
Processed 5400000 tokens.written 100000 lines
          11100000 tokens.written 200000 lines
          16700000 tokens.written 300000 lines
          22300000 tokens.written 400000 lines
          28000000 tokens.written 500000 lines
          33600000 tokens.written 600000 lines
          39200000 tokens.written 700000 lines
          44900000 tokens.written 800000 lines
          50600000 tokens.written 900000 lines
          56200000 tokens.written 1000000 lines
          61800000 tokens.written 1100000 lines
          67500000 tokens.written 1200000 lines
          73100000 tokens.written 1300000 lines
          78800000 tokens.written 1400000 lines
          84400000 tokens.written 1500000 lines
          90000000 tokens.written 1600000 lines
          95700000 tokens.written 1700000 lines
          101300000 tokens.written 1800000 lines
          101800000 tokens.real 13.14
user 7.29
sys 0.13
           697139200  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
               53019  page reclaims
                  58  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
               18542  voluntary context switches
                 205  involuntary context switches
        143788611105  instructions retired
         29858293141  cycles elapsed
           106087384  peak memory footprint
Processed 101880768 tokens.
Counted 267734 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 56928.

BUILDING VOCABULARY
Processed 5400000 tokens.written 100000 lines
          11100000 tokens.written 200000 lines
          16700000 tokens.written 300000 lines
          22300000 tokens.written 400000 lines
          28000000 tokens.written 500000 lines
          33600000 tokens.written 600000 lines
          39200000 tokens.written 700000 lines
          44900000 tokens.written 800000 lines
          50600000 tokens.written 900000 lines
          56200000 tokens.written 1000000 lines
          61800000 tokens.written 1100000 lines
          67500000 tokens.written 1200000 lines
          73100000 tokens.written 1300000 lines
          78800000 tokens.written 1400000 lines
          84400000 tokens.written 1500000 lines
          90000000 tokens.written 1600000 lines
          95700000 tokens.written 1700000 lines
          101300000 tokens.written 1800000 lines
          101800000 tokens.real 12.81
user 7.52
sys 0.14
           696090624  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
               52951  page reclaims
                  44  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
               18018  voluntary context switches
                 157  involuntary context switches
        143863412190  instructions retired
         30391919268  cycles elapsed
           106644416  peak memory footprint
Processed 101880768 tokens.
Counted 267734 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 56928.

BUILDING VOCABULARY
Processed 5400000 tokens.written 100000 lines
          11100000 tokens.written 200000 lines
          16700000 tokens.written 300000 lines
          22300000 tokens.written 400000 lines
          28000000 tokens.written 500000 lines
          33600000 tokens.written 600000 lines
          39200000 tokens.written 700000 lines
          44900000 tokens.written 800000 lines
          50600000 tokens.written 900000 lines
          56200000 tokens.written 1000000 lines
          61800000 tokens.written 1100000 lines
          67500000 tokens.written 1200000 lines
          73100000 tokens.written 1300000 lines
          78800000 tokens.written 1400000 lines
          84400000 tokens.written 1500000 lines
          90000000 tokens.written 1600000 lines
          95700000 tokens.written 1700000 lines
          101300000 tokens.written 1800000 lines
          101800000 tokens.real 12.98
user 7.43
sys 0.13
           697597952  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
               53043  page reclaims
                  44  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
               18137  voluntary context switches
                 203  involuntary context switches
        143835921381  instructions retired
         29952513910  cycles elapsed
           108217328  peak memory footprint
Processed 101880768 tokens.
Counted 267734 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 56928.

BUILDING VOCABULARY
Processed 5400000 tokens.written 100000 lines
          11100000 tokens.written 200000 lines
          16700000 tokens.written 300000 lines
          22300000 tokens.written 400000 lines
          28000000 tokens.written 500000 lines
          33600000 tokens.written 600000 lines
          39200000 tokens.written 700000 lines
          44900000 tokens.written 800000 lines
          50600000 tokens.written 900000 lines
          56200000 tokens.written 1000000 lines
          61800000 tokens.written 1100000 lines
          67500000 tokens.written 1200000 lines
          73100000 tokens.written 1300000 lines
          78800000 tokens.written 1400000 lines
          84400000 tokens.written 1500000 lines
          90000000 tokens.written 1600000 lines
          95700000 tokens.written 1700000 lines
          101300000 tokens.written 1800000 lines
          101800000 tokens.real 14.15
user 7.54
sys 0.13
           697761792  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
               53058  page reclaims
                  44  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
               18855  voluntary context switches
                 325  involuntary context switches
        143965422664  instructions retired
         30348723296  cycles elapsed
           108381168  peak memory footprint
Processed 101880768 tokens.
Counted 267734 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 56928.

BUILDING VOCABULARY
Processed 5400000 tokens.written 100000 lines
          11100000 tokens.written 200000 lines
          16700000 tokens.written 300000 lines
          22300000 tokens.written 400000 lines
          28000000 tokens.written 500000 lines
          33600000 tokens.written 600000 lines
          39200000 tokens.written 700000 lines
          44900000 tokens.written 800000 lines
          50600000 tokens.written 900000 lines
          56200000 tokens.written 1000000 lines
          61800000 tokens.written 1100000 lines
          67500000 tokens.written 1200000 lines
          73100000 tokens.written 1300000 lines
          78800000 tokens.written 1400000 lines
          84400000 tokens.written 1500000 lines
          90000000 tokens.written 1600000 lines
          95700000 tokens.written 1700000 lines
          101300000 tokens.written 1800000 lines
          101800000 tokens.real 13.53
user 7.40
sys 0.13
           695844864  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
               52933  page reclaims
                  44  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
               18612  voluntary context switches
                 180  involuntary context switches
        143622269597  instructions retired
         29774612366  cycles elapsed
           104825816  peak memory footprint
Processed 101880768 tokens.
Counted 267734 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 56928.

```


And then doing a clean build and rerun but with the buffering bump up ...


BUILDING VOCABULARY
Processed 5500000 tokens.written 100000 lines
          11100000 tokens.written 200000 lines
          16700000 tokens.written 300000 lines
          22300000 tokens.written 400000 lines
          28000000 tokens.written 500000 lines
          33600000 tokens.written 600000 lines
          39200000 tokens.written 700000 lines
          44900000 tokens.written 800000 lines
          50600000 tokens.written 900000 lines
          56200000 tokens.written 1000000 lines
          61800000 tokens.written 1100000 lines
          67500000 tokens.written 1200000 lines
          73100000 tokens.written 1300000 lines
          78800000 tokens.written 1400000 lines
          84400000 tokens.written 1500000 lines
          90000000 tokens.written 1600000 lines
          95700000 tokens.written 1700000 lines
          101300000 tokens.written 1800000 lines
          101800000 tokens.        7.94 real         7.24 user         0.15 sys
Processed 101879111 tokens.
Counted 267734 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 56928.


BUILDING VOCABULARY
Processed 5500000 tokens.written 100000 lines
          11100000 tokens.written 200000 lines
          16700000 tokens.written 300000 lines
          22300000 tokens.written 400000 lines
          28000000 tokens.written 500000 lines
          33600000 tokens.written 600000 lines
          39200000 tokens.written 700000 lines
          44900000 tokens.written 800000 lines
          50600000 tokens.written 900000 lines
          56200000 tokens.written 1000000 lines
          61800000 tokens.written 1100000 lines
          67500000 tokens.written 1200000 lines
          73100000 tokens.written 1300000 lines
          78800000 tokens.written 1400000 lines
          84400000 tokens.written 1500000 lines
          90000000 tokens.written 1600000 lines
          95700000 tokens.written 1700000 lines
          101300000 tokens.written 1800000 lines
          101800000 tokens.        7.84 real         7.13 user         0.14 sys
Processed 101879111 tokens.
Counted 267734 unique words.
Truncating vocabulary at min count 50.
Using vocabulary of size 56928.



Now it looks like the two tokens are at:

> DEBUG_WORD 'Fats' seen at token index 203761335
> DEBUG_WORD 'Fats' seen at token index 203761383

We print a small window around the first index:

python3 stream.py | sed -e 's/<unk>/<raw_unk>/g' | tr -s '[:space:]' '\n' | awk 'NR>=203761320 && NR<=203761350 {printf "%9d\t%s\n", NR, $0}'

Eg 15 tokens before and after the first missing




FAST
/usr/bin/time
3.73 real         0.00 user         0.07 sys

COUNTING COOCCURRENCES
window size: 15
context: symmetric
max product: 13752509
overflow length: 38028356
Reading vocab from file "vocab.txt"...loaded 56928 words.
Building lookup table...table contains 88816867 elements.
Processed 101880768 tokens.
Writing cooccurrences to disk.........2 files in total.
Merging cooccurrence files: processed 1 lines.

ORIG

/usr/bin/time
 12.20 real         0.00 user         0.08 sys

COUNTING COOCCURRENCES
window size: 15
context: symmetric
max product: 13752509
overflow length: 38028356
Reading vocab from file "vocab.txt"...loaded 56928 words.
Building lookup table...table contains 88816867 elements.
Processed 101880768 tokens.
Writing cooccurrences to disk.........2 files in total.
Merging cooccurrence files: processed 1 lines.


./build/cooccur -read cooccur_orig.bin -print | sort > orig.txt
/build/cooccur -read cooccur_fastreader.bin -print | sort > fast.txt

diff -u orig.txt fast.txt



# full workflow-text8 validation
```
./build/vocab_count -min-count 0 < text8 > vocab.txt
 ./build/cooccur -verbose 2 -vocab-file vocab.txt -memory 4 -overflow-file OF <text8
mv OF_merged.bin orig_OF_merged.bin # or `fast_*` depending on the build you use
diff {fast,orig}_OF_merged.bin # these are not different!!!
```
fastReader branch

COUNTING COOCCURRENCES
window size: 15
context: symmetric
max product: 13752509
overflow length: 38028356
Reading vocab from file "vocab.txt"...loaded 253854 words.
Building lookup table...table contains 129739125 elements.
Processed 17005207 tokens.
Writing cooccurrences to disk..........3 files in total.
Merging cooccurrence files: processed 71976140 lines.

master branch

COUNTING COOCCURRENCES
window size: 15
context: symmetric
max product: 13752509
overflow length: 38028356
Reading vocab from file "vocab.txt"...loaded 253854 words.
Building lookup table...table contains 129739125 elements.
Processed 17005207 tokens.
Writing cooccurrences to disk..........3 files in total.
Merging cooccurrence files: processed 71976140 lines.


OK let us look at the wikitext-103-v1 too. This is a more messy corpus and would expose issues likely.

#  full workflow wikitext-103-v1 validation

```sh
python3 stream.py | sed -e 's/<unk>/<raw_unk>/g' | ./build/vocab_count -min-count 50 -verbose 2 > vocab.txt
python3 stream.py | sed -e 's/<unk>/<raw_unk>/g' | ./build/cooccur -verbose 2 -vocab-file vocab.txt -memory 4 -overflow-file OF
mv OF_merged.bin orig_OF_merged.bin # or fast_* if on that branch..
diff {fast,orig}_OF_merged.bin # also not different...
```

# for timings
```sh
/usr/bin/time ./build/vocab_count -min-count 10 < /tmp/corpus.tokens > vocab.txt
/usr/bin/time ./build/cooccur -verbose 2 -vocab-file vocab.txt -memory 4 -overflow-file OF < /tmp/corpus.tokens
./build/shuffle -memory 4 -verbose 2 < OF_merged.bin > OF_shuffled.bin
./build/glove -save-file vectors -threads 4 -input-file OF_shuffled.bin -vocab-file vocab.txt -vector-size 50 -iter 50 -x-max 100 -binary 2 -verbose 2 -eta 0.05
```

# Evaluations
For some evals using a few common datasets and a custom script

```sh
python3 word_sim_eval.py --wordsim353-path eval/wordsim353/combined.csv \
                         --MEN-path eval/MEN/MEN_dataset_natural_form_full \
                         --MC-folder  eval/Miller_and_Charles/ \
                         --vectors vectors.txt
```

# GloVe-V

To quantify the uncertainty in the glove embeddings you can extract data to construct a variance-covariance
matrix using eqns 6 and 7 from the 2024 NAACL paper [GloVe-V](https://aclanthology.org/2024.emnlp-main.510/).

There is code for this as well but this is still WIP and hasn't been used to conduct any evaluations
or comparisons with fastread vectors.

The commits on the `master` with the GloVe-V calculations are: cb7d31d and 4853982
which were made on Jan 23rd and Jan 30th respectively.

*TODOs:*

1. Tidy up code is `misc.py` (and give it a better name) and compare to the variances and variance-covariance matrices
that are returned via the GloVe-V paper.

2. Add notes describing the steps to generate the vector data necessary from the `src` executables. Write a worked example.
