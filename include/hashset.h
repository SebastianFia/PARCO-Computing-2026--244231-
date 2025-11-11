#ifndef __HASHSET_H__
#define __HASHSET_H__

#include <stdlib.h>
#include <stdio.h>

typedef struct node_t {
    int val;
    struct node_t* next;
} node_t;

typedef struct hashset_t {
    int num_buckets;
    int num_elements;
    node_t** buckets_array;
} hashset_t;

hashset_t* hashset_create(int num_buckets);

int hash(int val, int num_buckets);

void hashset_insert(hashset_t* set, int val);

int hashset_contains(const hashset_t* set, int val);

void hashset_remove(hashset_t* set, int val);

void hashset_free(hashset_t* set);

#endif
