import nengo
import numpy as np

# Adapted from Nengo Networks
from nengo.networks import EnsembleArray


def IA(d, n_neurons, dt, share_thresholding_intercepts=False):
    bar_beta = 2.  # should be >= 1 + max_input * tau2 / tau1
    tau_model1 = 0.1
    tau_model2 = 0.1
    tau_actual = 0.1

    # dynamics put into continuous LTI form:
    #   dot{x1} = A1x1 + A2x2 + Bu
    # where x1 is the state variable for layer 1 and
    #       x2 is the state variable for layer 2
    # note that from the perspective of Principle 3, A2x2 is treated
    # as an "input" similar to u
    I = np.eye(d)
    inhibit = 1 - I
    B = 1. / tau_model1  # input -> layer 1
    A1 = 0  # (integrator) layer1 -> layer1
    A2 = (I - bar_beta * inhibit) / tau_model2  # layer 2 -> layer 1

    n_neurons_threshold = 50  
    n_neurons_x = n_neurons - n_neurons_threshold
    assert n_neurons_x > 0
    threshold = 0.8

    with nengo.Network(label="IA") as net:
        net.input = nengo.Node(size_in=d)
        x = nengo.networks.EnsembleArray(
            n_neurons_x, d,
            eval_points=nengo.dists.Uniform(0., 1.),
            intercepts=nengo.dists.Uniform(0., 1.),
            encoders=nengo.dists.Choice([[1.]]), label="Layer 1")
        net.x = x
        nengo.Connection(x.output, x.input, transform=tau_actual * A1 + I,
                         synapse=tau_actual)

        nengo.Connection(
            net.input, x.input,
            transform=tau_actual * B,
            synapse=tau_actual)

        with nengo.presets.ThresholdingEnsembles(0.):
            thresholding = nengo.networks.EnsembleArray(
                n_neurons_threshold, d, label="Layer 2")
            if share_thresholding_intercepts:
                for e in thresholding.ensembles:
                    e.intercepts = nengo.dists.Exponential(
                        0.15, 0., 1.).sample(n_neurons_threshold)
            net.output = thresholding.add_output('heaviside', lambda x: x > 0.)

        bias = nengo.Node(1., label="Bias")

        nengo.Connection(x.output, thresholding.input, synapse=0.005)
        nengo.Connection(
            bias, thresholding.input, transform=-threshold * np.ones((d, 1)))
        nengo.Connection(
            thresholding.heaviside, x.input,
            transform=tau_actual * A2, synapse=tau_actual)

    return net


def InputGatedMemory(n_neurons, n_neurons_diff, dimensions, feedback=1.0,
                     difference_gain=1.0, recurrent_synapse=0.1,
                     difference_synapse=None, net=None, **kwargs):
    """Stores a given vector in memory, with input controlled by a gate.
    Parameters
    ----------
    n_neurons : int
        Number of neurons per dimension in the vector.
    dimensions : int
        Dimensionality of the vector.
    feedback : float, optional (Default: 1.0)
        Strength of the recurrent connection from the memory to itself.
    difference_gain : float, optional (Default: 1.0)
        Strength of the connection from the difference ensembles to the
        memory ensembles.
    recurrent_synapse : float, optional (Default: 0.1)
    difference_synapse : Synapse (Default: None)
        If None, ...
    kwargs
        Keyword arguments passed through to ``nengo.Network``.
    Returns
    -------
    net : Network
        The newly built memory network, or the provided ``net``.
    Attributes
    ----------
    net.diff : EnsembleArray
        Represents the difference between the desired vector and
        the current vector represented by ``mem``.
    net.gate : Node
        With input of 0, the network is not gated, and ``mem`` will be updated
        to minimize ``diff``. With input greater than 0, the network will be
        increasingly gated such that ``mem`` will retain its current value,
        and ``diff`` will be inhibited.
    net.input : Node
        The desired vector.
    net.mem : EnsembleArray
        Integrative population that stores the vector.
    net.output : Node
        The vector currently represented by ``mem``.
    net.reset : Node
        With positive input, the ``mem`` population will be inhibited,
        effectively wiping out the vector currently being remembered.
    """
    if net is None:
        kwargs.setdefault('label', "Input gated memory")
        net = nengo.Network(**kwargs)
    else:
        warnings.warn("The 'net' argument is deprecated.", DeprecationWarning)

    if difference_synapse is None:
        difference_synapse = recurrent_synapse

    n_total_neurons = n_neurons * dimensions
    n_total_neurons_diff = n_neurons_diff * dimensions

    with net:
        # integrator to store value
        
        mem_net = nengo.Network()
        mem_net.config[nengo.Ensemble].encoders = nengo.dists.Choice([[-1.]]) 
        mem_net.config[nengo.Ensemble].radius = 1
        mem_net.config[nengo.Ensemble].eval_points=nengo.dists.Uniform(-1, 0.0) 
        mem_net.config[nengo.Ensemble].intercepts = nengo.dists.Uniform(-0.6, 1.) 
        
    
        with mem_net:
            net.mem = EnsembleArray(n_neurons, dimensions, label="mem")
        nengo.Connection(net.mem.output, net.mem.input,
                         transform=feedback,
                         synapse=recurrent_synapse)

        
        diff_net = nengo.Network()
        diff_net.config[nengo.Ensemble].radius = 0.5
        diff_net.config[nengo.Ensemble].eval_points=nengo.dists.Uniform(-0.5, 0.5)
        with diff_net:
            # calculate difference between stored value and input
            net.diff = EnsembleArray(n_neurons_diff, dimensions, label="diff")
        nengo.Connection(net.mem.output, net.diff.input, transform=-1)

        # feed difference into integrator
        nengo.Connection(net.diff.output, net.mem.input,
                         transform=difference_gain,
                         synapse=difference_synapse)

        # gate difference (if gate==0, update stored value,
        # otherwise retain stored value)
        net.gate = nengo.Node(size_in=1)
        net.diff.add_neuron_input()
        nengo.Connection(net.gate, net.diff.neuron_input,
                         transform=np.ones((n_total_neurons_diff, 1)) * -10,
                         synapse=None)

        # reset input (if reset=1, remove all values, and set to 0)
        net.reset = nengo.Node(size_in=1)
        nengo.Connection(net.reset, net.mem.add_neuron_input(),
                         transform=np.ones((n_total_neurons, 1)) * -3,
                         synapse=None)

    net.input = net.diff.input
    net.output = net.mem.output

    return net