import argparse

args = argparse.Namespace(
    niters=500,  # Number of iterations for training--------------------------------------
    F=4,  # The disaggregation number
    dec_res_dim=1,  # Dimensionality after decoding
    seq_l=504,  # window of preprocessing data for GRU
    input_dims=6,
    conditions ="no_window_decoder_to_1dim",

    n=504,  # Size of the dataset----------------------------------------

    lr=1e-2,  # Starting learning rate-----------------------------------
    batch_size=128,  # Batch size for training--------------------------
    latent_ode=True,  # Run Latent ODE seq2seq model --------------------------------------------
    viz=True,  # Show plots while training----------------------------------

    dataset='periodic',  # Dataset to load. Available: physionet, activity, hopper, periodic----------------------------
    z0_encoder='rnn',  # Type of encoder for Latent ODE model: odernn or rnn------------------------------------
    rec_dims=20,  # Dimensionality of the recognition model (ODE or RNN)---------------------------------------
    rec_layers=1,  # Number of layers in ODE function in recognition ODE------------------------------
    gen_layers=1,  # Number of layers in ODE function in generative ODE-----------------------------
    units=100,  # Number of units per layer in ODE function-----------------------------------------
    gru_units=100,  # Number of units per layer in each of GRU update networks-----------------------
    latent_dims=6,  # Size of the latent state------------------------------------
    timepoints=500,  # Total number of time-points------------------------
    max_t=5.,  # We subsample points in the interval [0, args.max_t]-----------------------------

    save='experiments/',  # Path for saving checkpoints
    random_seed=1991,  # Random seed for reproducibility
    sample_tp=None,
    # Number of time points to sub-sample. If > 1, subsample exact number of points. If the number is in [0,1],
    # take a percentage of available points per time series. If None, do not subsample
    cut_tp=None,
    # Cut out the section of the timeline of the specified length (in number of points). Used for periodic function demo
    quantization=0.1,
    # Quantization on the physionet dataset. Value 1 means quantization by 1 hour, value 0.1 means quantization by
    # 0.1 hour = 6 min
    classic_rnn=False,
    # Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only
    rnn_cell='gru',  # RNN Cell type. Available: gru (default), expdecay
    input_decay=False,
    # For RNN: use the input that is the weighted average of empirical mean and previous value (like in GRU-D)
    ode_rnn=False,  # Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only
    rnn_vae=False,  # Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss
    poisson=False,  # Model poisson-process likelihood for the density of events in addition to reconstruction
    classif=False,  # Include binary classification loss -- used for Physionet dataset for hospital mortality
    linear_classif=False,  # If using a classifier, use a linear classifier instead of 1-layer NN
    extrap=False,  # Set extrapolation mode. If this flag is not set, run interpolation mode
    noise_weight=0.01  # Noise amplitude for generated trajectories
)
