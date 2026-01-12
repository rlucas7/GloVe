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

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include <stdio.h>

/* default FastReader buffer size (tunable) */
#ifndef FR_DEFAULT_BUFSIZE
#define FR_DEFAULT_BUFSIZE (1<<16) /* 64 KiB */
#endif

#ifdef _MSC_VER
#define STRERROR(ERRNO, BUF, BUFSIZE) strerror_s((BUF), (BUFSIZE), (ERRNO))
#else
#define STRERROR(ERRNO, BUF, BUFSIZE) strerror_r((ERRNO), (BUF), (BUFSIZE))
#endif

/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while (*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return (*s1 - *s2);
}

/* Move-to-front hashing and hash function from Hugh Williams, http://www.seg.rmit.edu.au/code/zwh-ipl/ */

/* Simple bitwise hash function */
unsigned int bitwisehash(char *word, int tsize, unsigned int seed) {
    char c;
    unsigned int h;
    h = seed;
    for ( ; (c = *word) != '\0'; word++) h ^= ((h << 5) + c + (h >> 2));
    return (unsigned int)((h & 0x7fffffff) % tsize);
}

/* Create hash table, initialise pointers to NULL */
HASHREC ** inithashtable(void) {
    int i;
    HASHREC **ht;
    ht = (HASHREC **) malloc( sizeof(HASHREC *) * TSIZE );
    for (i = 0; i < TSIZE; i++) ht[i] = (HASHREC *) NULL;
    return ht;
}

/* ---------- FastReader: instance-based buffered reader (reentrant) ---------- */
/* Usage:
     FastReader fr;
     if (fastreader_init(&fr, stdin, FR_DEFAULT_BUFSIZE) == 0) {
         while (!get_word_fast(buf, &fr)) { ... }
         fastreader_destroy(&fr);
     } else {
         fallback...
     }
*/

/* Initialize FastReader instance. Returns 0 on success, -1 on malloc failure */
int fastreader_init(FastReader *fr, FILE *f, size_t bufsz) {
    if (!fr) return -1;
    fr->f = f;
    fr->buf = malloc(bufsz);
    if (!fr->buf) return -1;
    fr->bufsz = bufsz;
    fr->idx = fr->len = 0;
    fr->pushed = 0;
    fr->pushch = 0;
    return 0;
}

/* Tear down FastReader (free buffer). Call when done with reader. */
void fastreader_destroy(FastReader *fr) {
    if (!fr) return;
    if (fr->buf) free(fr->buf);
    fr->buf = NULL;
    fr->f = NULL;
    fr->idx = fr->len = 0;
    fr->pushed = 0;
}

/*If EOF is encountered, determine whether tokens still requiring
  processing exist on the buffer.  */
int fastreader_eof(FastReader *fr) {
    if (!fr || !fr->f) return 1; /* treat as EOF on error */
    /* Underlying FILE* EOF flag may be set when fread hit EOF even if fr->buf
       still contains bytes. Only report EOF when there are no unread bytes
       in the buffer and no pushed byte is pending. */
    if (!feof(fr->f)) return 0;
    if (fr->pushed) return 0;
    if (fr->idx < fr->len) return 0; /* still buffered bytes */
    return 1;
}


/* Like fgetc but reads from the FastReader */
static inline int fastreader_getc(FastReader *fr) {
    if (fr->pushed) {
        int c = (unsigned char)fr->pushch;
        fr->pushed = 0;
        // DEBUG code
        //fprintf(stderr, "[FR] getc -> PUSHED '%c' (0x%02x)\n", (c >= 32 && c < 127) ? c : '?', c);
        return c;
    }
    if (fr->idx >= fr->len) {
       //DEBUG  size_t prev_len = fr->len;
        fr->len = fread(fr->buf, 1, fr->bufsz, fr->f);
        fr->idx = 0;
        // DEBUG code
        // fprintf(stderr, "[FR] refill: read=%zu prev_len=%zu bufsz=%zu\n", fr->len, prev_len, fr->bufsz);
        if (fr->len == 0){
            // DEBUG code
            //fprintf(stderr, "[FR] getc -> EOF\n");
            return EOF;
        }
    }
    int ch = (unsigned char)fr->buf[fr->idx++];
    //DEBUG code
    //fprintf(stderr, "[FR] getc -> BUF '%c' (0x%02x) idx=%zu len=%zu\n", (ch >= 32 && ch < 127) ? ch : '?', ch, fr->idx, fr->len);
    return ch;
}

/* one-char pushback (for newline handling) */
static inline void fastreader_ungetc(FastReader *fr, int ch) {
    fr->pushed = 1;
    fr->pushch = ch & 0xFF;

    // DEBUG code
    //fprintf(stderr, "[FR] ungetc -> PUSHED '%c' (0x%02x)\n", (ch >= 32 && ch < 127) ? ch : '?', ch & 0xFF);
}

/* Reentrant get_word that uses a FastReader instance.
   Returns 1 when encounter '\n' or EOF (but separate from word), 0 otherwise. */
int get_word_fast(char *word, FastReader *fr) {
    int i = 0, ch;
    if (!fr || !fr->f) return 0; /* defensive */

    for (;;) {
        ch = fastreader_getc(fr);
        if (ch == '\r') continue;
        if (i == 0 && ((ch == '\n') || (ch == EOF))) {
            word[i] = 0;
            return 1;
        }
        if (i == 0 && ((ch == ' ') || (ch == '\t'))) continue; // skip leading space
        if ((ch == EOF) || (ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (ch == '\n') fastreader_ungetc(fr, ch);
            break;
        }
        if (i < MAX_STRING_LENGTH - 1)
            word[i++] = ch; // don't allow words to exceed MAX_STRING_LENGTH
    }
    word[i] = 0; // null terminate

    /* preserve existing truncation safety for UTF-8 multi-byte sequences */
    if (i == MAX_STRING_LENGTH - 1 && (word[i-1] & 0x80) == 0x80) {
        if ((word[i-1] & 0xC0) == 0xC0) {
            word[i-1] = '\0';
        } else if (i > 2 && (word[i-2] & 0xE0) == 0xE0) {
            word[i-2] = '\0';
        } else if (i > 3 && (word[i-3] & 0xF8) == 0xF0) {
            word[i-3] = '\0';
        }
    }
    return 0;
}

/* Read word from input stream. Return 1 when encounter '\n' or EOF (but separate from word), 0 otherwise.
   Words can be separated by space(s), tab(s), or newline(s). Carriage return characters are just ignored.
   (Okay for Windows, but not for Mac OS 9-. Ignored even if by themselves or in words.)
   A newline is taken as indicating a new document (contexts won't cross newline).
   Argument word array is assumed to be of size MAX_STRING_LENGTH.
   words will be truncated if too long. They are truncated with some care so that they
   cannot truncate in the middle of a utf-8 character, but
   still little to no harm will be done for other encodings like iso-8859-1.
   (This function appears identically copied in vocab_count.c and cooccur.c.)
*/

/* Backward-compatible wrapper for single-threaded code:
   Uses a static FastReader on first call (for convenience). This wrapper is NOT thread-safe;
   for threaded readers, create a FastReader per-thread and call get_word_fast(). */
int get_word(char *word, FILE *fin) {
    static FastReader fr_static;
    static int initialized = 0;
    if (!initialized) {
        if (fastreader_init(&fr_static, fin, FR_DEFAULT_BUFSIZE) == 0) {
            initialized = 1;
        } else {
            /* allocation failed; fallback to original fgetc-based implementation */
            fr_static.f = NULL;
        }
    }

    if (initialized && fr_static.f == fin) {
        return get_word_fast(word, &fr_static);
    }

    /* Fallback: original fgetc-based loop (unchanged) */
    int i = 0, ch;
    for (;;) {
        ch = fgetc(fin);
        if (ch == '\r') continue;
        if (i == 0 && ((ch == '\n') || (ch == EOF))) {
            word[i] = 0;
            return 1;
        }
        if (i == 0 && ((ch == ' ') || (ch == '\t'))) continue; // skip leading space
        if ((ch == EOF) || (ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (ch == '\n') ungetc(ch, fin); // return the newline next time as document ender
            break;
        }
        if (i < MAX_STRING_LENGTH - 1)
          word[i++] = ch; // don't allow words to exceed MAX_STRING_LENGTH
    }
    word[i] = 0; //null terminate
    // avoid truncation destroying a multibyte UTF-8 char except if only thing on line (so the i > x tests won't overwrite word[0])
    // see https://en.wikipedia.org/wiki/UTF-8#Description
    if (i == MAX_STRING_LENGTH - 1 && (word[i-1] & 0x80) == 0x80) {
        if ((word[i-1] & 0xC0) == 0xC0) {
            word[i-1] = '\0';
        } else if (i > 2 && (word[i-2] & 0xE0) == 0xE0) {
            word[i-2] = '\0';
        } else if (i > 3 && (word[i-3] & 0xF8) == 0xF0) {
            word[i-3] = '\0';
        }
    }
    return 0;
}

/* ---------- remainder of original common.c follows unchanged ---------- */

int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if (!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

void free_table(HASHREC **ht) {
    int i;
    HASHREC* current;
    HASHREC* tmp;
    for (i = 0; i < TSIZE; i++) {
        current = ht[i];
        while (current != NULL) {
            tmp = current;
            current = current->next;
            free(tmp->word);
            free(tmp);
        }
    }
    free(ht);
}

void free_fid(FILE **fid, const int num) {
    int i;
    for(i = 0; i < num; i++) {
        if(fid[i] != NULL)
            fclose(fid[i]);
    }
    free(fid);
}


int log_file_loading_error(char *file_description, char *file_name) {
    fprintf(stderr, "Unable to open %s %s.\n", file_description, file_name);
    fprintf(stderr, "Errno: %d\n", errno);
    char error[MAX_STRING_LENGTH];
    STRERROR(errno, error, MAX_STRING_LENGTH);
    fprintf(stderr, "Error description: %s\n", error);
    return errno;
}
