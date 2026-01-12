#ifndef COMMON_H
#define COMMON_H

//  Common code for cooccur.c, vocab_count.c,
//  glove.c and shuffle.c
//
//  GloVe: Global Vectors for Word Representation
//  Copyright (c) 2014 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    Christopher Manning (manning@cs.stanford.edu)
//    https://github.com/stanfordnlp/GloVe/
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/

#include <stdio.h>

#define MAX_STRING_LENGTH 1000
#define TSIZE 1048576
#define SEED 1159241
#define HASHFN bitwisehash

/* Default FastReader buffer size (tunable). Adjust as needed. */
#ifndef FR_DEFAULT_BUFSIZE
#define FR_DEFAULT_BUFSIZE (1<<16) /* 64 KiB */
#endif

typedef double real;
typedef struct cooccur_rec {
    int word1;
    int word2;
    real val;
} CREC;
typedef struct hashrec {
    char *word;
    long long num; //count or id
    struct hashrec *next;
} HASHREC;


/* Forward-declare FastReader type and I/O helper functions for buffered reading
   FastReader: instance-based buffered reader used by get_word_fast().
   Put the struct here so callers (e.g., vocab_count.c) can allocate instances. */
typedef struct FastReader {
    FILE *f;
    char *buf;
    size_t bufsz;
    size_t idx;   /* next read index */
    size_t len;   /* bytes valid in buf */
    int pushed;   /* one-byte pushback flag */
    int pushch;   /* value for pushback */
} FastReader;

int fastreader_init(FastReader *fr, FILE *f, size_t bufsz);
void fastreader_destroy(FastReader *fr);
int get_word_fast(char *word, FastReader *fr);

/* Return non-zero when the underlying FILE* is at EOF and the FastReader has
   no unread buffered bytes and no pushed byte pending. Use to decide whether
   the reader has truly exhausted the input. Used when the stream is less than
   the set buffer size to ensure all tokens are processed and premature exit
   is not executed. */
int fastreader_eof(FastReader *fr);

int scmp( char *s1, char *s2 );
unsigned int bitwisehash(char *word, int tsize, unsigned int seed);
HASHREC **inithashtable(void);
int get_word(char *word, FILE *fin);
void free_table(HASHREC **ht);
int find_arg(char *str, int argc, char **argv);
void free_fid(FILE **fid, const int num);

// logs errors when loading files.  call after a failed load
int log_file_loading_error(char *file_description, char *file_name);

#endif /* COMMON_H */

