#pragma once

#include <stddef.h>
#include <stdbool.h>

#ifndef PRELU_ALPHA
#define PRELU_ALPHA 0.005
#endif

#define CML_FILE_MAGIC 0x4D4C4D43 /* CMLM */

enum cml_activation {
    CML_ACT_NONE,
    CML_ACT_SIGMOID,
    CML_ACT_PRELU
};

struct cml_context {
    size_t max_alloc;
    size_t cur_alloc;
};

struct cml_layer {
    struct cml_layer *next;
    struct cml_layer *prev;
    
    int count;
    int wcount;

    float *data;
    float *bias;
    float *weights;

    float *delta;

    enum cml_activation actfn;
};

#ifdef  __cplusplus
extern "C" {
#endif

struct cml_layer *cml_new_layer(struct cml_context *ctx, struct cml_layer **root, int count, enum cml_activation activation);
void cml_free(struct cml_context *ctx, struct cml_layer **root);

bool cml_save_model(char *path, struct cml_layer **root);
bool cml_load_model(struct cml_context *ctx, char *path, struct cml_layer **root);

struct cml_layer *cml_randomize_layer(struct cml_layer *layer, bool bias);

void cml_fwd(struct cml_layer *root);
void cml_bwd(struct cml_context *ctx, struct cml_layer *root, float *last_layer_raw, float step);

#ifdef  __cplusplus
}
#endif