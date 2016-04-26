-- Some helper methods for AdaBoost
-- Computes pseudoloss, samples data while assigning weights, etc.
require 'rubiks'

function _q(prob_output, right_label, wrong_label)
    return 0.5 * (1 - prob_output[right_label] + prob_output[wrong_label])
end

function _ploss(prob_output, right_label)
    -- Pseudoloss for one example
    -- Defined as 0.5 * (1 - Pr[correct label] + \sum_{wrong} weight_wrong * Pr[wrong])
    --   where weight_wrong sums to 1
    -- Pseudoloss is 1 when Pr[correct] = 0
    -- Pseudoloss is 0 when Pr[correct] = 1
    -- When output is the uniform distribution, pseudoloss = 0.5
    --
    -- Thus we can use 1/2 - pseudoloss as the edge, just like 2 class AdaBoost
    -- AdaBoost does a thing where specific mis-classifications are weighted more,
    -- here just use weight_wrong = 1 / (K-1)
    --
    -- We are allowed to output hypotheses with outputs in [0,1]
    -- Right now, the output is a probability distribution, so even
    -- when the argmax is correct, the pseudoloss margin below 0.5
    -- can be pretty low.
    -- We scale the output to emphasis the gap
    local total = 1
    local cop = prob_output:clone()
    local largest, _ = prob_output:max(1)
    largest = largest[1]
    cop = cop / largest

    local output = torch.zeros(N_MOVES)
    if CUDA then
        output = output:cuda()
    end

    for class = 1, N_MOVES do
        if class != right_label then
            output[class] = _q(output, right_label, class)
        end
    end
    return output
end


function pseudoloss(episode_outputs, labels, n_episodes, episode_length)
    -- Given weak learner outputs, this gives the pseudoloss for the weak learner
    -- This outputs the average pseudoloss over the episode
    -- Weighting of pseudoloss will be done elsewhere
    local losses = torch.Tensor(n_episodes * episode_lengths, N_MOVES):zero()
    if CUDA then
        losses = losses:cuda()
    end

    for i = 1, n_episodes * episode_length do
        -- Feed samples to get average pseudoloss
        local ploss = _ploss(episode_outputs[i], labels[i])
        losses[ep] = ploss
    end
    return losses
end


function nextWeights(weights, losses, alpha, total_weight)
    -- Next weight prop to D_t(i, y) * exp(alpha_t * (pseudoloss(i,y) + 1))
    weights = weights:clone()
    local factor = (losses + 1) * alpha
    local next_w = weights:cmul(factor:exp())
    return next_w * total_weight / next_w:sum()
end


function computeBoostedWeightsAndAcc(model, episodes, labels, n_episodes, episode_length, weight_to_replace)
    -- Model is expected to be an Averager instance (outputs
    -- probabilities from averaged models)
    local outputs = torch.Tensor(n_episodes * episode_length, N_MOVES):zero()
    if CUDA then
        outputs = outputs:cuda()
    end

    -- Compute in one enormous batch
    local seqIndices = torch.LongTensor():range(
        1, 1 + (n_episodes - 1) * episode_length, episode_length
    )
    local inputs = {}
    for step = 1, episode_length do
        inputs[step] = episodes:index(1, seqIndices)
        seqIndices = seqIndices + 1
    end
    model:forget()
    local _outputs = model:forward(inputs)
    -- Drop into right locations
    local seqIndices = torch.LongTensor():range(
        1, 1 + (n_episodes - 1) * episode_length, episode_length
    )
    for step = 1, episode_length do
        outputs:indexCopy(1, seqIndices, _outputs[step])
        seqIndices = seqIndices + 1
    end

    -- Compute the accuracy of the boosted model while we're here
    _, best = outputs:max(2)
    best:resize(n_episodes * episode_length)
    correct = labels:eq(best):sum()

    -- Final derivation has weight work out to boosted pseudoloss + 1 (assuming weights
    -- in boosted are normalized)
    local losses = pseudoloss(outputs, labels, n_episodes, episode_length)
    losses = losses + 1
    local weights = losses:exp()
    -- Set weights to 0 for correct
    for i = 1, n_episodes * episode_length do
        weights[i][labels[i]] = 0
    end
    -- Normalize to amount of weight to replace
    local normed = weights * weight_to_replace / weights:sum()
    return normed, correct
end
