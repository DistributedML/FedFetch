
def is_batch_norm_layer(key):
    return ("num_batches_tracked" in key) or ('running' in key)