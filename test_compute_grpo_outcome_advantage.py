# %%
import torch
import numpy as np
from collections import defaultdict

def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   length_penalty: str = "v1",
                                   length_penalty_alpha: float = 0.5,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        length_penalty: `(str)`
            length penalty type, can be "v1" or "v2"
        length_penalty_alpha: `(float)`
            alpha value for length penalty
        epsilon: `(float)`
            epsilon value for length penalty
    
    Returns:
        scores: `(torch.Tensor)`
            shape: (bs, response_length)
        scores: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    if length_penalty == "v3" or length_penalty == "v4":
        return compute_grpo_outcome_advantage_lp_new(token_level_rewards=token_level_rewards, eos_mask=eos_mask, index=index, length_penalty=length_penalty, length_penalty_alpha=length_penalty_alpha, epsilon=epsilon)
    
    print("length_penalty:", length_penalty, "length_penalty_alpha:", length_penalty_alpha)
    
    # For debugging
    # torch.save({
    #     'token_level_rewards': token_level_rewards,
    #     'eos_mask': eos_mask,
    #     'index': index,
    #     'length_penalty': length_penalty,
    #     'length_penalty_alpha': length_penalty_alpha,
    #     'epsilon': epsilon
    # }, f'compute_grpo_outcome_advantage.pth')
    
    # token_level_rewards: only the last token will have reward
    # For example, if token_level_rewards[index, 161] = 1, the eos_mask.sum(dim=1)[index] should be 162, and all other entries in token_level_rewards[index] should be 0
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    scores_with_length_penalty = scores.clone()
    actual_length = eos_mask.sum(dim=-1)

    id2score = defaultdict(list)
    id2length = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append((scores[i], i))
            id2length[index[i]].append(actual_length[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # length penalty is not applied
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                # length penalty is applied
                # shape: (bs,)
                current_scores = torch.tensor([score for score, _ in id2score[idx]])
                
                if length_penalty:
                    assert torch.all((current_scores == 0) | (current_scores == 1)), "current_scores must be 0 or 1 if length penalty is applied"
                    current_length = torch.tensor(id2length[idx])
                    max_length, min_length = current_length.max(), current_length.min()
                    l = length_penalty_alpha * (0.5 - (current_length - min_length + epsilon / 2) / (max_length - min_length + epsilon))
                    
                    if length_penalty == "v1":
                        current_scores = current_scores + torch.where(current_scores > 0, l, torch.min(l, torch.tensor(0.)))
                    elif length_penalty == "v2":
                        # v2: only apply length penalty to the scores that are greater than 0
                        current_scores = current_scores + torch.where(current_scores > 0, l, torch.zeros_like(l))
                    else:
                        raise ValueError(f"unknown length penalty type: {length_penalty}")
                    # print("current_length:", current_length, "l:", l, "current_scores:", current_scores)
                    for i in range(len(id2score[idx])):
                        # print(f"before scores_with_length_penalty ({i, id2score[idx][i]}):", scores_with_length_penalty[id2score[idx][i][1]], "current_scores:", current_scores[i])
                        scores_with_length_penalty[id2score[idx][i][1]] = current_scores[i]

                id2mean[idx] = torch.mean(current_scores)
                id2std[idx] = torch.std(current_scores[None])
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        # print("scores_with_length_penalty:", scores_with_length_penalty[0], "scores:", scores[0])
        scores = scores_with_length_penalty
        for i in range(bsz):
            # print(i, index[i], scores[i], id2mean[index[i]], id2std[index[i]], id2score[index[i]])
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            # print(i, index[i], scores[i])
            # raise ValueError("stop here")
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

def compute_grpo_outcome_advantage_old(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   length_penalty: bool = False,
                                   length_penalty_alpha: float = 0.5,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    if length_penalty:
        print("length penalty is not supported in the original GRPO implementation")
    
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

def compute_grpo_outcome_advantage_lp_new(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   length_penalty: str = "v1",
                                   length_penalty_alpha: float = 0.5,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        length_penalty: `(str)`
            length penalty type, can be "v1" or "v2"
        length_penalty_alpha: `(float)`
            alpha value for length penalty
        epsilon: `(float)`
            epsilon value for length penalty
    
    Returns:
        scores: `(torch.Tensor)`
            shape: (bs, response_length)
        scores: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    
    print("length_penalty:", length_penalty, "length_penalty_alpha:", length_penalty_alpha)
    
    # For debugging
    # torch.save({
    #     'token_level_rewards': token_level_rewards,
    #     'eos_mask': eos_mask,
    #     'index': index,
    #     'length_penalty': length_penalty,
    #     'length_penalty_alpha': length_penalty_alpha,
    #     'epsilon': epsilon
    # }, f'compute_grpo_outcome_advantage.pth')
    
    # token_level_rewards: only the last token will have reward
    # For example, if token_level_rewards[index, 161] = 1, the eos_mask.sum(dim=1)[index] should be 162, and all other entries in token_level_rewards[index] should be 0
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    # This is just the length penalty scores, not the final scores
    length_penalty_scores = torch.zeros_like(scores)
    actual_length = eos_mask.sum(dim=-1)

    id2score = defaultdict(list)
    id2length = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append((scores[i], i))
            id2length[index[i]].append(actual_length[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # length penalty is not applied
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                # length penalty is applied
                # shape: (bs,)
                current_scores = torch.tensor([score for score, _ in id2score[idx]])
                current_length = torch.tensor([length for length in id2length[idx]])
                
                if length_penalty:
                    assert torch.all((current_scores == 0) | (current_scores == 1)), "current_scores must be 0 or 1 if length penalty is applied"
                    
                    if length_penalty == "v3":
                        max_length, min_length = current_length.max(), current_length.min()
                        l = length_penalty_alpha * (0.5 - (current_length - min_length + epsilon / 2) / (max_length - min_length + epsilon))
                        # v2 and v3: only apply length penalty to the scores that are greater than 0
                        current_length_penalty = torch.where(current_scores > 0, l, torch.zeros_like(l))
                    elif length_penalty == "v4":
                        # only compute max and min length for the scores that are greater than 0
                        current_length_filtered = torch.tensor([length for length, score in zip(current_length, current_scores) if score > 0])
                        if len(current_length_filtered) == 0:
                            # no length penalty is applied since all scores are 0
                            current_length_penalty = torch.zeros_like(current_scores)
                        else:
                            max_length, min_length = current_length_filtered.max(), current_length_filtered.min()
                            # Here we still use current_length to preserve the shape of the scores (the correct mask will be applied, and items with score 0 will be masked out)
                            l = length_penalty_alpha * (0.5 - (current_length - min_length + epsilon / 2) / (max_length - min_length + epsilon))
                            # v2 and v3: only apply length penalty to the scores that are greater than 0
                            current_length_penalty = torch.where(current_scores > 0, l, torch.zeros_like(l))

                            # print("current_length_filtered:", current_length_filtered, current_length_filtered.max(), current_length_filtered.min(), "idx:", idx)
                    else:
                        raise ValueError(f"unknown length penalty type: {length_penalty}")
                    # print("current_length:", current_length, "current_length_penalty:", current_length_penalty)
                    for i in range(len(id2score[idx])):
                        # print(f"before scores_with_length_penalty ({i, id2score[idx][i]}):", scores_with_length_penalty[id2score[idx][i][1]], "current_length_penalty:", current_length_penalty[i])
                        length_penalty_scores[id2score[idx][i][1]] = current_length_penalty[i]

                id2mean[idx] = torch.mean(current_scores)
                id2std[idx] = torch.std(current_scores[None])
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        # print("scores_with_length_penalty:", scores_with_length_penalty[0], "scores:", scores[0])
        for i in range(bsz):
            # print(i, index[i], scores[i], id2mean[index[i]], id2std[index[i]], id2score[index[i]])
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon) + length_penalty_scores[i]
            # print(i, index[i], scores[i])
            # raise ValueError("stop here")
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


# %%
info = torch.load('compute_grpo_outcome_advantage_0.pth')
print(info)
token_level_rewards, eos_mask, index, epsilon = info['token_level_rewards'], info['eos_mask'], info['index'], info['epsilon']
length_penalty = "v4"
length_penalty_alpha = 0.1

# indices_mask = torch.tensor(np.array(index) == index[0])
correct_num = 4
found = False
for i in range(len(index)):
    indices_mask = torch.tensor(np.array(index) == index[i])
    if token_level_rewards[indices_mask].sum().item() == correct_num:
        print("Using index:", index[i])
        found = True
        break
if not found:
    raise ValueError("No index found")

print("index:", index[indices_mask])

print("token_level_rewards (summed with last dim):", token_level_rewards[indices_mask].sum(dim=-1))

scores, _ = compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index, length_penalty, length_penalty_alpha, epsilon)
print("scores with indices mask applied:", scores[indices_mask])

# scores, _ = compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index, length_penalty, length_penalty_alpha, epsilon)
# print("scores indices mask:", scores[indices_mask])

scores_old, _ = compute_grpo_outcome_advantage_old(token_level_rewards, eos_mask, index, length_penalty, length_penalty_alpha, epsilon)
print("scores old with indices mask applied:", scores_old[indices_mask])

# assert torch.allclose(scores, scores_old)


# %%
