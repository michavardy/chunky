import numpy as np
from hmmx.models import  HMM, HiddenStates, Observations
from hmmx.emissions.gaussian import GaussianEmission
from hmmx.transitions.discrete import DiscreteTransition
from hmmx.initial.discrete import DiscreteInitial

def inference_chunks(model: HMM, diffs: list[float]) -> list[int]:
    observations = Observations(sequence=diffs)
    hidden_states_sequence = model.inference(observations)
    return hidden_states_sequence

def get_auto_chunk_hmm_model(diffs: list[float]) -> HMM:
    hidden_states = HiddenStates(categories=["new_chunk", "prior_chunk"])
    observations = Observations(sequence=diffs)
    initial = DiscreteInitial(hidden_states, [0.25, 0.75])
    transition = DiscreteTransition(hidden_states, [[0.1, 0.9], [0.5, 0.5]])
    emmisions = GaussianEmission(
        hidden_states,
        mus=[np.percentile(diffs, 75), np.percentile(diffs, 25)],
        sigmas=[np.std(diffs), np.std(diffs)]
    )
    model = HMM(hidden_states, initial, transition, emmisions)
    model.fit(observations.sequence)
    return model