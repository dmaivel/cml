#include "cml.h"

#include <stdio.h>
#include <string.h>

#define make_color(r, g, b) r / 255.f, g / 252.f, b / 255.f
#define n_inputs 3
#define n_outputs 9
#define n_sets 9

float training_inputs[n_sets * n_inputs] = 
{ 
    1.f, 0.f, 0.f,
    1.f, .6f, 0.f,
    1.f, 1.f, 0.f,
    0.f, 1.f, 0.f,
    0.f, 0.f, 1.f,
    1.f, 0.f, 1.f,
    .4f, 0.f, .4f,
    1.f, 1.f, 1.f,
    0.f, 0.f, 0.f,
};

/* 
 * encoding:
 * red, orange, yellow, green, blue, pink, purple, white, black 
 */
float training_outputs[n_sets * n_outputs] = 
{ 
    1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f,
    0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f
};

int main(int argc, char **argv)
{
    struct cml_context ctx = {
        .max_alloc = 1048,
        .cur_alloc = 0
    };

    struct cml_layer *color_identifier = NULL;
    struct cml_layer *output;

    bool model_status = cml_load_model(&ctx, "color_model.cml", &color_identifier);

    if (!model_status) {
        printf("model not found, falling back on training...\n");
        cml_new_layer(&ctx, &color_identifier, n_inputs, CML_ACT_NONE);
        
        /* 
         * randomize hidden & output weights (only randomize output bias)
         */
        cml_randomize_layer(cml_new_layer(&ctx, &color_identifier, 12, CML_ACT_PRELU), false);
        output = cml_randomize_layer(cml_new_layer(&ctx, &color_identifier, n_outputs, CML_ACT_SIGMOID), true);
    } 
    else {
        printf("model found.\n");
        output = color_identifier->next->next;
    }

    printf("%ld/%ld bytes used\n", ctx.cur_alloc, ctx.max_alloc);

    /*
     * train network
     */
    if (!model_status) {
        for (int e = 0; e < 10000; e++) {
            for (int i = 0; i < n_sets; i++) {
                memcpy(color_identifier->data, &training_inputs[i * n_inputs], sizeof(float) * n_inputs);
                cml_fwd(color_identifier);
                cml_bwd(&ctx, color_identifier, &training_outputs[i * n_outputs], 0.5);
            }
        }
    }

    /*
     * test network w/ new input
     */
    float data[] = { make_color(54, 247, 118) };
    memcpy(color_identifier->data, data, sizeof(float) * n_inputs);
    cml_fwd(color_identifier);

    /*
     * print inputs
     */
    for (int i = 0; i < n_inputs; i++) {
        printf("\nin[%d] = %f", i, color_identifier->data[i]);
    }
    puts("");

    /*
     * print outputs
     */
    for (int i = 0; i < n_outputs; i++) {
        printf("\nout[%d] = %f", i, output->data[i]);
    }
    puts("");

    /*
     * determine prediction
     */
    int i = 0;
    for (int j = 0; j < n_outputs; j++) {
        if (output->data[j] > output->data[i])
            i = j;
    }

    printf("prediction: ");
    switch (i) {
    case 0: printf("red\n"); break;
    case 1: printf("orange\n"); break;
    case 2: printf("yellow\n"); break;
    case 3: printf("green\n"); break;
    case 4: printf("blue\n"); break;
    case 5: printf("pink\n"); break;
    case 6: printf("purple\n"); break;
    case 7: printf("white\n"); break;
    case 8: printf("black\n"); break;
    }

    if (!model_status)
        cml_save_model("color_model.cml", &color_identifier);
    cml_free(&ctx, &color_identifier);

    return 0;
}