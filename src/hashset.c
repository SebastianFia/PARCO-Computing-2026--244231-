#include <stdlib.h>
#include <stdio.h>

#include "hashset.h"

hashset_t *hashset_create(int num_buckets)
{
    if (num_buckets <= 0)
    {
        fprintf(stderr, "Num buckets must be > 0\n");
        return NULL;
    }

    hashset_t *set = malloc(sizeof(hashset_t));
    if (set == NULL)
    {
        perror("Failed to allocate set");
        return NULL;
    }

    set->buckets_array = calloc(num_buckets, sizeof(node_t *)); // init to NULLs
    if (set->buckets_array == NULL)
    {
        perror("Failed to allocate buckets_array in set");
        free(set);
        return NULL;
    }

    set->num_buckets = num_buckets;
    set->num_elements = 0;

    return set;
}

int hash(int val, int num_buckets)
{
    // With this logic we also handle negative vals
    return (val % num_buckets + num_buckets) % num_buckets;
}

void hashset_insert(hashset_t *set, int val)
{
    if (set == NULL || set->buckets_array == NULL)
    {
        fprintf(stderr, "Got NULL in insert function.\n");
        return;
    }

    // Find the bucket index at which we will find the val (if it is in the set)
    int index = hash(val, set->num_buckets);

    // Create new node
    node_t *new = malloc(sizeof(node_t));
    if (new == NULL)
    {
        perror("Failed to alloc new node, so set insert failed");
        return;
    }
    new->next = NULL;
    new->val = val;

    // If the list is empty, simply insert and return
    node_t *curr = set->buckets_array[index];
    if (curr == NULL)
    {
        set->buckets_array[index] = new;
        set->num_elements++;
        return;
    }

    while (1)
    {
        // If the value is already in the set, we do nothing and return
        if (curr->val == val)
        {
            free(new);
            return;
        }
        // When we reach the last node, we insert and return
        if (curr->next == NULL)
        {
            curr->next = new;
            set->num_elements++;
            return;
        }
        curr = curr->next;
    };
}

int hashset_contains(const hashset_t *set, int val)
{
    if (set == NULL || set->buckets_array == NULL)
    {
        fprintf(stderr, "Got NULL in get function.\n");
        return 0;
    }

    // Find the bucket index at which we will find the val (if it is in the set)
    int index = hash(val, set->num_buckets);

    node_t *curr = set->buckets_array[index];
    while (curr != NULL)
    {
        // If the value is in the set, we return 1
        if (curr->val == val)
            return 1;
        curr = curr->next;
    };

    // If no match found, we return 0
    return 0;
}

void hashset_remove(hashset_t *set, int val)
{
    if (set == NULL || set->buckets_array == NULL)
    {
        fprintf(stderr, "Got NULL in remove function.\n");
        return;
    }

    // Find the bucket index at which we will find the val (if it is in the set)
    int index = hash(val, set->num_buckets);

    node_t *curr = set->buckets_array[index];

    // If the list of the bucket is empty, we do nothing and return
    if (curr == NULL)
        return;

    // Iterate through the list of the bucket and remove the val if we find it
    node_t *prev = NULL;
    while (curr != NULL)
    {
        // If we find the value, we remove it and return
        if (curr->val == val)
        {
            if (prev == NULL)
                // If curr is the head, set a new head for the list
                set->buckets_array[index] = curr->next;
            else
                prev->next = curr->next;

            set->num_elements--;
            free(curr);
            return;
        }
        prev = curr;
        curr = curr->next;
    };
}

void hashset_free(hashset_t *set)
{
    if (set == NULL || set->buckets_array == NULL)
        return;

    for (int i = 0; i < set->num_buckets; i++)
    {
        node_t *next;
        node_t *curr = set->buckets_array[i];
        while (curr != NULL)
        {
            next = curr->next;
            free(curr);
            curr = next;
        }
    }
    free(set->buckets_array);
    free(set);
}