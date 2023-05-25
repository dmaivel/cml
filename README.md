# cml
Basic neural network implementation written in C.
## build demo
```bash
git clone https://github.com/dmaivel/cml
cd cml
make
```
## features
- CPU only
- Save/load models (custom implementation)
- Lightweight
## usage
For a complete demo, refer to `demo.c`.
### initialization
```c
struct cml_context ctx = {
    .max_alloc = 1048, /* max amount of bytes program may allocate */ 
    .cur_alloc = 0 /* current allocation (should be 0) */ 
};
```
### creating models
```c
// network becomes root
struct cml_layer *network = NULL;
struct cml_layer *input_layer = cml_new_layer(&ctx, &network, n_inputs, CML_ACT_NONE);
struct cml_layer *hidden_layer = cml_new_layer(&ctx, &network, 12, CML_ACT_PRELU);
...
struct cml_layer *output_layer = cml_new_layer(&ctx, &network, 10, CML_ACT_SIGMOID);

// randomize data inside layer
cml_randomize_layer(hidden_layer, false); // won't randomize bias
cml_randomize_layer(output_layer, true); // will randomize bias
```
### reading & writing inputs/outputs
```c
// single input (network may be substituted w/ `input_layer`)
network->data[...] = ...;

// entire input layer
memcpy(network->data, raw_data, sizeof(float) * network->count);

// single output
float x = output_layer->data[...];

// entire output layer
memcpy(raw_data, output_layer->data, sizeof(float) * output_layer->count);
```
### passes
```c
// forward pass
cml_fwd(network);

// backwards prop (allocates memory, so context needs to be passed)
cml_bck(&ctx, network, &raw_training_outputs[...], step_size);
```
### loading/saving models
```c
// load model
struct cml_layer *network = NULL;
cml_load_model(&ctx, "model_name.cml", &network);

// save model
cml_save_model("model_name.cml", &network);
```
### free models
```c
cml_free(&ctx, &network);
```