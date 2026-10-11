"""Teacher-forced action metrics with the official causal/vision-token alignment."""
import numpy as np


def teacher_forced_metrics(logits, labels, tokenizer, statistics, loss):
    # Prismatic inserts vision tokens after BOS. This is the same shift used by
    # the pinned finetune.py: logits[:, num_patches:-1] vs labels[:, 1:].
    patches = logits.shape[1] - labels.shape[1]
    if patches <= 0 or labels.shape[0] != 1:
        raise ValueError('Expected one multimodal example with inserted vision tokens')
    targets = labels[:, 1:]
    mask = targets > tokenizer.action_token_begin_idx
    if int(mask.sum()) != 7 or int((labels != -100).sum()) != 8:
        raise ValueError('Expected seven action labels plus one supervised stop token')
    predicted = logits[:, patches:-1].argmax(dim=-1)[mask].detach().cpu().numpy()
    target = targets[mask].detach().cpu().numpy()
    normalized = tokenizer.decode_token_ids_to_actions(predicted)
    target_normalized = tokenizer.decode_token_ids_to_actions(target)
    lo, hi = np.asarray(statistics['q01']), np.asarray(statistics['q99'])
    action = np.where(statistics['mask'], .5*(normalized+1)*(hi-lo)+lo, normalized)
    # Official decoding clips arbitrary token IDs to action bins. Expose these
    # cases rather than treating the clipped command as a valid action token.
    invalid = (predicted < tokenizer.tokenizer.vocab_size-tokenizer.n_bins) | (predicted >= tokenizer.tokenizer.vocab_size)
    result = dict(action=action.tolist(), predicted_token_ids=predicted.tolist(),
                  target_token_ids=target.tolist(), token_matches=(predicted == target).tolist(),
                  action_token_accuracy=float(np.mean(predicted == target)),
                  normalized_l1_token_targets=float(np.mean(np.abs(normalized-target_normalized))),
                  loss_including_stop=float(loss), supervised_tokens=8,
                  invalid_action_tokens=invalid.tolist())
    if not np.isfinite(action).all() or not np.isfinite(result['loss_including_stop']):
        raise ValueError('Nonfinite teacher-forced output')
    return result
