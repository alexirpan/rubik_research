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
    local total = 1
    for class = 1, N_MOVES do
        if class == right_label then
            total = total - prob_output[right_label]
        else
            total = total + prob_output[wrong_label] / (N_MOVES - 1)
        end
    end
    return 0.5 * total
end


function pseudoloss(episode_outputs, labels, weights, n_episodes, episode_length)
    -- Given weak learner outputs, this gives the pseudoloss over distribution
    -- D_t for the weak learner.
    local loss = 0
    -- prob_outputs is a table here
    for ep = 1, n_episodes do
        local episode = episode_outputs[ep]
        local start = (ep - 1) * episode_length
        for step = 1, episode_length do
            local ind = start + step
            local correct = labels[ind]
            loss = loss + weights[ep] * _ploss(episode[step], correct)
        end
    end
    -- weights are on a per episode basis, but each is episode_length samples
    local norm = weights:sum() * episode_length
    -- this is average pseudoloss
    return loss / norm
end


function regenerateDataset(model, weak_learner_outputs, alpha, prev_data, prev_labels, prev_weights, n_episodes, episode_length)
    -- Takes the previous weak learner outputs on the previous dataset.
    -- Resamples from the previous data. Reweights the newly generated data
    local eps = torch.Tensor(n_episodes * episode_length, N_STICKERS * N_COLORS):zero()
    local eps_labels = torch.LongTensor(n_episodes * episode_length):zero()
    local weights = torch.Tensor(n_episodes):zero()
    -- Horribly abusing Lua globals here - this should be set
    -- in time
    if CUDA then
        eps = eps:cuda()
        eps_labels = eps_labels:cuda()
        weights = weights:cuda()
    end

    local j = 1
    for i = 1, n_episodes do
        -- Resample from last epoch
        -- exp(alpha) if misclassified
        -- TODO
    end
    while j <= n_episodes do
        local episode, moves = randomCubeEpisode(episode_length)
        -- Horribly abusing Lua globals here - this should be set
        -- in time
        if CUDA then
            episode = episode:cuda()
            moves = moves:cuda()
        end
        episode:resize(episode_length, N_STICKERS * N_COLORS)
        total_samples = total_samples + 1
        local prob = accept_prob(model, episode, moves, episode_length, target_error, dt_prime)
        if weighted then
            episode_weights[i] = prob
        end
        if weighted or torch.uniform() < prob then
            local start = (i-1) * episode_length
            eps[{ {start+1, start+episode_length} }] = episode
            eps_labels[{ {start+1, start+episode_length} }] = moves
            i = i + 1
            n_fails = 0
            -- the algorithm design gives more time if samples are accepted
            dt_prime = confidence / (i * (i+1))
            threshold = _threshold(target_error, dt_prime, i)
        else
            n_fails = n_fails + 1
            if n_fails >= threshold then
                print('Failed to generate, wrapping up training')
                return
            end
        end
        if total_samples % 100000 == 0 then
            print(total_samples, 'episodes sampled for epoch so far')
            print(i, 'episodes accepted for dataset')
            print(math.ceil(threshold), 'rejects in a row needed to stop')
        end
    end
    if weighted then
        return eps, eps_labels, episode_weights
    else
        return eps, eps_labels, total_samples
    end
end


function estimateEdge(outputs, labels, n_episodes, episode_length)
    -- This is passed the run of model outputs from higher up the stack
    -- Exponentiate outputs
    for step = 1, episode_length do
        outputs[step] = outputs[step]:exp()
    end
    -- Shapes:
    --  outputs is a table of epslen elements, each is n_test x N_MOVES
    --  labels is labels for ep 1, ep 2, ep3, all concatenated together
    --  The first episode is the first entry in each table tensor
    --  The second is the 2nd entry in each table tensor
    --  etc.
    local episode_outputs = {}
    for ep = 1, n_episodes do
        episode_outputs[ep] = torch.Tensor(episode_length, N_MOVES)
        -- Again, abusing Lua globals here
        if CUDA then
            episode_outputs[ep] = episode_outputs[ep]:cuda()
        end

        for step = 1, episode_length do
            episode_outputs[ep][step] = outputs[step][ep]
        end
    end
    return 0.5 - pseudoloss(episode_outputs, labels, n_episodes, episode_length)
end
