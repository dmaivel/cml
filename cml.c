#include "cml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

static void *cml_alloc_wrapper(struct cml_context *ctx, size_t size)
{
    assert(ctx->cur_alloc + size < ctx->max_alloc);
    ctx->cur_alloc += size;
    return calloc(1, size);
}

static void cml_free_wrapper(struct cml_context *ctx, void *ptr, size_t size)
{
    assert(ctx->cur_alloc - size >= 0);
    ctx->cur_alloc -= size;
    free(ptr);
}

static void *cml_list_alloc(struct cml_context *ctx, void **root)
{
    /*
     * if list hasn't started, then start the dam list
     */
    if (*root == NULL) {
        *root = cml_alloc_wrapper(ctx, sizeof(struct cml_layer));
        return *root;
    }

    /*
     * otherwise, iterate through the list until we find the next spot to allocate to
     */
    void *next;

    /*
     * iterate through root until next = NULL
     */
    for (next = *root; *(void**)(next); next = *(void**)(next));

    /* 
     * allocation 
     */
    *(void**)(next) = cml_alloc_wrapper(ctx, sizeof(struct cml_layer));
    return *(void**)(next);
}

struct cml_layer *cml_new_layer(struct cml_context *ctx, struct cml_layer **root, int count, enum cml_activation activation)
{
    struct cml_layer *result = cml_list_alloc(ctx, (void**)root);

    struct cml_layer *prev;
    for (prev = *root; prev; prev = prev->next)
        if (prev->next == result)
            break;
    result->prev = prev;

    result->count = count;
    result->data = cml_alloc_wrapper(ctx, count * sizeof(float));
    result->bias = cml_alloc_wrapper(ctx, count * sizeof(float));
    result->weights = cml_alloc_wrapper(ctx, count * (prev ? prev->count : 0) * sizeof(float));
    result->actfn = activation;
    result->wcount = count * (prev ? prev->count : 0);

    return result;
}

void cml_free(struct cml_context *ctx, struct cml_layer **root)
{
    /* 
     * recursively call up until the last element 
     * (elements need to be freed in reverse order) 
     */
    if (*(void**)(root)) {
        cml_free(ctx, *(void**)(root));

        /*
         * multiply by 2 to account for delta 
         */
        ctx->cur_alloc -= (*root)->count * sizeof(float) * 2 + (*root)->wcount * sizeof(float);
        cml_free_wrapper(ctx, *root, sizeof(struct cml_layer));
    }
}

bool cml_save_model(char *path, struct cml_layer **root)
{
    /* 
     * open file, write magic
     */
    FILE *f = fopen(path, "wb");
    if (f == NULL)
        return false;

    int magic = CML_FILE_MAGIC;
    fwrite(&magic, sizeof(magic), 1, f);

    /* 
     * write layer count 
     */
    int layer_c = 0;
    for (struct cml_layer *layer = *root; layer; layer = layer->next)
        layer_c++;
    fwrite(&layer_c, sizeof(layer_c), 1, f);

    /*
     * write layers
     */
    for (struct cml_layer *layer = *root; layer; layer = layer->next) {
        fwrite(&layer->actfn, sizeof(layer->actfn), 1, f);
        fwrite(&layer->count, sizeof(layer->count), 1, f);
        
        fwrite(layer->data, layer->count * sizeof(float), 1, f);
        fwrite(layer->bias, layer->count * sizeof(float), 1, f);
        fwrite(layer->weights, layer->wcount * sizeof(float), 1, f);
    }

    fclose(f);
    return true;
}

bool cml_load_model(struct cml_context *ctx, char *path, struct cml_layer **root)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL)
        return false;
    
    /*
     * check magic
     */
    int magic;
    fread(&magic, sizeof(magic), 1, f);
    if (magic != CML_FILE_MAGIC)
        return false;
    
    /* 
     * get layer count
     */
    int layer_c;
    fread(&layer_c, sizeof(layer_c), 1, f);
    
    /* 
     * loop thru layers 
     */
    for (int i = 0; i < layer_c; i++) {
        enum cml_activation act;
        int count;

        fread(&act, sizeof(act), 1, f);
        fread(&count, sizeof(count), 1, f);
        
        struct cml_layer *layer = cml_new_layer(ctx, root, count, act);
        fread(layer->data, count * sizeof(float), 1, f);
        fread(layer->bias, count * sizeof(float), 1, f);
        fread(layer->weights, layer->wcount * sizeof(float), 1, f);
    }

    fclose(f);
    return true;
}

struct cml_layer *cml_randomize_layer(struct cml_layer *layer, bool bias)
{
    if (bias)
        for (int i = 0; i < layer->count; i++)
            layer->bias[i] = ((float)rand()) / (float)RAND_MAX;

    for (int i = 0; i < layer->count; i++)
        layer->data[i] = ((float)rand()) / (float)RAND_MAX;
    for (int i = 0; i < layer->wcount; i++)
        layer->weights[i] = ((float)rand()) / (float)RAND_MAX;
    
    return layer;
}

static float cml_activation_impl(enum cml_activation act, float x)
{
    switch (act) {
    case CML_ACT_SIGMOID:
        return 1 / (1 + expf(-x));
    case CML_ACT_PRELU:
        return MAX(0, x) + (PRELU_ALPHA * MIN(0, x));
    default:
        return x;
    };
}

static float cml_activation_impl_d(enum cml_activation act, float x)
{
    switch (act) {
    case CML_ACT_SIGMOID:
        return x * (1 - x);
    case CML_ACT_PRELU:
        return PRELU_ALPHA;
    default:
        return x;
    };
}

void cml_fwd(struct cml_layer *root)
{
    assert(root->next && "cannot complete forward pass on single layer");

    struct cml_layer *prev = root;
    struct cml_layer *curr;

    for (curr = root->next; curr; curr = curr->next) {
        for (int i = 0; i < curr->count; i++) {
            float activation = curr->bias[i];

            /* summation */
            for (int j = 0; j < prev->count; j++) {
                activation += prev->data[j] * curr->weights[j * curr->count + i];
            }

            curr->data[i] = cml_activation_impl(curr->actfn, activation);
        }

        prev = curr;
    }
}

void cml_bwd(struct cml_context *ctx, struct cml_layer *root, float *last_layer_raw, float step)
{
    assert(root->next && "cannot complete backward pass on single layer");

    struct cml_layer *last;
    for (last = root; last->next != NULL; last = last->next);

    struct cml_layer *prev = NULL;
    struct cml_layer *curr;

    /*
     * compute delta
     */
    for (curr = last; curr && curr != root; curr = curr->prev) {
        curr->delta = cml_alloc_wrapper(ctx, curr->count * sizeof(float));

        for (int i = 0; i < curr->count; i++) {
            if (prev == NULL)
                curr->delta[i] = (last_layer_raw[i] - curr->data[i]) * cml_activation_impl_d(curr->actfn, curr->data[i]);
            else {
                float error = 0;

                /* summation */
                for (int j = 0; j < prev->count; j++) {
                    error += prev->delta[j] * prev->weights[i * prev->count + j];
                }

                curr->delta[i] = error * cml_activation_impl_d(curr->actfn, curr->data[i]);
            }
        }

        prev = curr;
    }

    /*
     * apply changes based on delta
     */
    for (curr = last; curr && curr != root; curr = curr->prev) {
        for (int i = 0; i < curr->count; i++) {
            curr->bias[i] += curr->delta[i] * step;

            /* summation */
            for (int j = 0; j < curr->prev->count; j++) {
                curr->weights[j * curr->count + i] += curr->prev->data[j] * curr->delta[i] * step;
            }
        }

        prev = curr;
    }
    
    /*
     * free information
     */
    for (curr = last; curr; curr = curr->prev) {
        if (curr->delta)
            cml_free_wrapper(ctx, curr->delta, curr->count * sizeof(float));
    }
}