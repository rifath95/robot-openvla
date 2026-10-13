"""Validated inference-only sampling controls and action-vocabulary filtering."""
import math


def generation_options(sample_actions=False, temperature=1.0, top_p=1.0):
    if type(sample_actions) is not bool:
        raise ValueError('sample_actions must be boolean')
    if (type(temperature) not in (int,float) or not math.isfinite(temperature) or temperature<=0 or
        type(top_p) not in (int,float) or not math.isfinite(top_p) or not 0<top_p<=1):
        raise ValueError('temperature must be positive and top_p must be in (0,1]')
    if not sample_actions:
        return {'do_sample':False}
    return dict(do_sample=True,temperature=float(temperature),top_p=float(top_p),top_k=0,num_beams=1)


def validate_options(options):
    if options is None:return generation_options()
    if not isinstance(options,dict) or set(options)-{'do_sample','temperature','top_p','top_k','num_beams'}:
        raise ValueError('Unsupported action generation options')
    if options.get('top_k',0)!=0 or options.get('num_beams',1)!=1:
        raise ValueError('Action sampling requires top_k=0 and num_beams=1')
    return generation_options(options.get('do_sample',False),options.get('temperature',1.),options.get('top_p',1.))


def add_sampling_arguments(parser):
    parser.add_argument('--sample-actions',action='store_true',help='Sample valid action tokens instead of greedy decoding')
    parser.add_argument('--temperature',type=float,default=1.,help='Sampling temperature; lower values concentrate on likely tokens')
    parser.add_argument('--top-p',type=float,default=1.,help='Sampling probability mass to retain, in (0,1]')


class ActionTokenLogitsProcessor:
    """Exclude text/EOS/padding tokens from the seven sampled action components."""
    def __init__(self,vocab_size,n_action_bins):
        if not 0<n_action_bins<vocab_size:raise ValueError('Invalid action token range')
        self.lower=vocab_size-n_action_bins
        self.upper=vocab_size

    def __call__(self,input_ids,scores):
        if scores.shape[-1]<self.upper:raise ValueError('Model vocabulary smaller than action token range')
        scores[:,:self.lower]=-float('inf')
        scores[:,self.upper:]=-float('inf')
        return scores
