#!/usr/bin/env python3
# -*- coding: utf-8 -*-

EXPERIMENT_CONFIG = {
    # number of configuration options (~configuration space complexity)
    "features": [50, 100, 250, 500],
    # spread of the influence distribution (Laplace distribution with mode 0)
    "effect_spread": [50],
    # repetition identifier (not the repetitions per data point) for generating different software systems
    "experiment_repetition_id": [0, 1, 2],
    # parameter controlling the interaction degree (distribution), parameter for the geometric distribution (higher p -> lower interaction degree)
    "p_interaction_degree": [0.6, 0.65, 0.7],
    # number of relevant terms (options and interactions) varied at three levels (see FEATURE_SIZE_CONFIG)
    "relevant_terms_level": [0, 1, 2],
}

FEATURE_SIZE_CONFIG = {
    # absolute number of relevant terms for different numbers of features (notice the Fibonacci sequence?:)
    "absolute_relevant_terms": {
        50: [3, 5, 8],
        100: [5, 8, 13],
        250: [8, 13, 21],
        500: [13, 21, 34],
    }
}

# number of repetitions per data point (software system and sample size)
#REPETITIONS = 30

# which sample sizes to test
SCREENING_CONFIG = {
    "relative_sample_size": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
}
