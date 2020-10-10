import brevitas


@brevitas.jit.script
def over_tensor(x):
    return (-1)


@brevitas.jit.script
def over_output_channels(x):
    return (x.shape[0], -1)


@brevitas.jit.script
def over_batch_over_tensor(x):
    return (x.shape[0], -1)


@brevitas.jit.script
def over_batch_over_output_channels(x):
    return (x.shape[0], x.shape[1], -1)