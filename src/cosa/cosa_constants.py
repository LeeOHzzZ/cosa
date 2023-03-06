#!/usr/bin/env python3 


# j=7, v=3, prob - var 
_A = [
    [1, 0, 0],  # R
    [1, 0, 0],  # S
    [0, 1, 1],  # P
    [0, 1, 1],  # Q
    [1, 1, 0],  # C
    [1, 0, 1],  # K
    [0, 1, 1],  # N
]

# A special treatment to adapt conv2d problem to linear layer
# set the C to be the input(reduction) dimension of Matrix Multiplication
# set the P to be the batch dimension
# set the Q to be the output dimension
# all other dimensions are unrelated
_A_FLEXASR = [
    [0, 0, 0], # R
    [0, 0, 0], # S
    [0, 1, 1], # P (batch dimension is related to both inputs and outputs)
    [1, 0, 1], # Q (output dimension is related to both weights and outputs)
    [1, 1, 0], # C (input dimension is related tp both weights and inputs)
    [0, 0, 0], # K
    [0, 0, 0], # N
]

# assume 6 levels of ranks
# v=3, i=6 var - rank
_B = [
    [1, 0, 1, 0, 0, 1],  # Weights
    [0, 0, 0, 1, 1, 1],  # Inputs
    [0, 1, 0, 0, 1, 1],  # Outputs
]

_B_HLSCNN = [
    [1, 1, 1],  # Weights
    [1, 1, 1],  # Inputs
    [1, 1, 1],  # Outputs
]

_B_FLEXASR = [
    [1, 0, 1, 1], # Weights
    [0, 1, 1, 1], # Inputs
    [0, 1, 1, 1], # Outputs
]

# for uneven mapping
# v=3, i=6, i'=6
_Z = [
    # Weights
    [
        [1, 0, 0, 0, 0, 0],  # mem 0
        [0, 0, 0, 0, 0, 0],  # mem 1
        [1, 1, 1, 0, 0, 0],  # mem 2
        [0, 0, 0, 0, 0, 0],  # mem 3
        [0, 0, 0, 0, 0, 0],  # mem 4
        [1, 1, 1, 1, 1, 1],  # mem 5
    ],
    # Inputs
    [
        [0, 0, 0, 0, 0, 0],  # mem 0
        [0, 0, 0, 0, 0, 0],  # mem 1
        [0, 0, 0, 0, 0, 0],  # mem 2
        [1, 1, 1, 1, 0, 0],  # mem 3
        [1, 1, 1, 1, 1, 0],  # mem 4
        [1, 1, 1, 1, 1, 1],  # mem 5
    ],
    # Outputs
    [
        [0, 0, 0, 0, 0, 0],  # mem 0
        [1, 1, 0, 0, 0, 0],  # mem 1
        [0, 0, 0, 0, 0, 0],  # mem 2
        [0, 0, 0, 0, 0, 0],  # mem 3
        [1, 1, 1, 1, 1, 0],  # mem 4
        [1, 1, 1, 1, 1, 1],  # mem 5
    ],
]
