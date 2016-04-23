-- Some helper methods for AdaBoost
-- Computes pseudoloss, samples data while assigning weights, etc.
require 'rubiks'

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
    local largest, best = prob_output:max(1)
    largest = largest[1]
    best = best[1]

    for class = 1, N_MOVES do
        if class == right_label then
            total = total - prob_output[class] / largest
        else
            total = total + prob_output[class] / (largest * (N_MOVES - 1))
        end
    end
    return 0.5 * total, best == right_label
end


function pseudoloss(episode_outputs, labels, n_episodes, episode_length)
    -- Given weak learner outputs, this gives the pseudoloss for the weak learner
    -- This outputs the average pseudoloss over the episode
    -- Weighting of pseudoloss will be done elsewhere
    local losses = torch.Tensor(n_episodes):zero()
    if CUDA then
        losses = losses:cuda()
    end
    local a = 0
    local b = 0
    local c = 0
    local d = 0

    for ep = 1, n_episodes do
        -- Pull out episode
        local start = (ep - 1) * episode_length
        local episode = episode_outputs[{ {start+1, start+episode_length} }]
        local labels = labels[{ {start+1, start+episode_length} }]
        -- Feed samples to get average pseudoloss
        for step = 1, episode_length do
            local ploss, is_right = _ploss(episode[step], labels[step])
            losses[ep] = losses[ep] + ploss
            if is_right then
                a = a + ploss
                b = b + 1
            else
                c = c + ploss
                d = d + 1
            end
        end
    end
    print('Average over good', a / b)
    print('Average over bad', c / d)
    print('Num good', b, 'Num bad', d)
    -- Norm over episode length
    return losses / episode_length
end


function nextWeights(weights, losses, alpha, n_episodes)
    -- Next weight prop to D_t(i) * exp(alpha_t * pseudoloss(i))
    weights = weights:clone()
    local factor = losses * alpha
    local next_w = weights:cmul(factor:exp())
    -- Normalize such that there are n_episodes eps total
    return next_w * n_episodes / next_w:sum()
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

    local losses = pseudoloss(outputs, labels, n_episodes, episode_length)
    local weights = losses:exp()
    -- Normalize to amount of weight to replace
    local normed = weights * weight_to_replace / weights:sum()
    return normed, correct
end
