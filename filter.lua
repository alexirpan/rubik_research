require 'rubiks'


function _qprime(prob_output, right_label, wrong_label)
    -- Un-normalized weight
    local diff = prob_output[right_label] - prob_output[wrong_label]
    return 1 / (1 + math.exp(0.5 * diff))
end


function _pprime(prob_output, right_label)
    -- Sum un-normalized weight
    local total = 0
    for class = 1, N_MOVES do
        if class ~= right_label then
            total = total + _qprime(prob_output, right_label, class)
        end
    end
    return total
end


function _qvec(prob_output, right_label)
    -- A Tensor of the normalized weights
    -- Always size 12, with one entry as zero
    local dist = torch.Tensor(12)
    for class = 1, N_MOVES do
        if class == right_label then
            dist[class] = 0
        else
            dist[class] = _qprime(prob_output, right_label, wrong_label)
        end
    end
    return dist / dist:sum()
end


function psuedoloss(prob_outputs, labels)
    -- Implements the pseudoloss function used in FilterBoost
    -- This assumes the output is the probability, not the log probability
    -- The pseudoloss is defined such that:
    -- - If Pr[correct label] = 0, the pseudoloss is 1
    -- - If Pr[correct label] = 1, the pseudoloss is 0
    -- - The uniform distribution has pseudoloss 0.5
    -- - Pr[correct label] > 1/k ==> pseudoloss < 0.5
    -- - The pseudoloss is larger if the model puts a large weight on an
    --   incorrect label
    --
    -- This lets us define edge by 1/2 - pseudoloss, and also makes the
    -- optimization prefer solutions that output one clear winning class

    -- We're rejecting on a per-episode basis instead of a per-sample
    -- basis, so average the pseudoloss over the samples
    -- TODO check theoretical guarantees of this
    local loss = 0
    for step = 1, #labels do
        local correct = labels[step]
        local weights = _qvec(prob_outputs[step], correct)
        -- The summation over wrong labels
        local sampleloss = weights * prob_outputs[step]
        -- Term for correct label
        sampleloss = 1 - prob_outputs[step][correct]
        loss = loss + 0.5 * sampleloss
    end
    return loss / #labels
end


function accept_prob(model, episode, labels, edge, goal_err, confidence)
    -- Returns the accept probability of the sample
    -- By returning the probability directly, we can let other algorithms
    -- decide whether to accept/reject or to weight the sample
    model:zeroGradParameters()
    model:forget()

    -- move episode to table
    local input = {}
    for step = 1, #episode do
        input[step] = episode[step]
    end
    local output = model:forward(input)
    -- Again use average over episode
    local avg_pprime = 0
    for step = 1, #episode do
        avg_pprime = avg_pprime + _pprime(output[step], labels[step])
    end
    avg_pprime = avg_pprime / #episode

    local prob = avg_pprime / (N_MOVES - 1)
    return prob
end


function _threshold(err, conf, prev_calls)
    return 2 / ((N_MOVES - 1) * err) * math.log(1 / conf)
end


function nextDataset(model, target_error, confidence, n_episodes, episode_length)
    -- FilterBoost halts when the error of the model (defined according to
    -- the pseudoloss) is <= target_error with probability >= 1 - confidence
    -- These essentially control the stopping criterion
    local eps = torch.Tensor(n_episodes * episode_length, N_STICKERS, N_COLORS):zero()
    local eps_labels = torch.LongTensor(n_episodes * episode_length):zero()

    local i = 1
    local total_samples = 0
    local n_fails = 0
    local dt_prime = confidence / 2
    local threshold = _threshold(target_error, dt_prime, i)

    while i <= n_episodes do
        local episode, moves = randomCubeEpisode(episode_length)
        total_samples = total_samples + 1
        local prob = accept_prob(model, episode, moves, FILL, FILL, FILL)
        if torch.uniform() < prob then
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
        if total_samples % 1000 == 0 then
            print(total_samples, 'episodes sampled for epoch so far')
            print(i, 'episodes accepted for dataset')
            print(threshold, 'rejects in a row needed to stop')
        end
    end
    return eps, eps_labels, total_samples
end


function estimateEdge(model, episodes, labels, episode_length)
    -- Following recommended implementation details, the number
    -- of samples used each round is fixed instead of adaptive. So,
    -- we can have this method take the dataset as the argument directly
    -- Episodes: seqlen * episodes x featsize
    -- Labels: episodes
    -- Since we're not interested in running updates we can just run
    -- the whole batch
    local n_episodes = labels:size(1)
    local seqIndices = torch.LongTensor():range(
        1, 1 + (n_episodes-1) * episode_length, episode_length
    )
    local inputs = {}
    for step = 1, episode_length do
        inputs[step] = episodes:index(1, seqIndices)
        seqIndices = seqIndices + 1
    end
    model:forget() -- forget past test runs
    local outputs = model:forward(inputs)
    -- (seqlen, n_episodes, N_MOVES)
    -- Exponentiate outputs
    for step = 1, episode_length do
        outputs[step] = outputs[step]:exp()
    end
    -- Rearrange back into output for the episodes
    local episode_outputs = {}
    for ep = 1, n_episodes do
        episode_outputs[ep] = {}
        for step = 1, episode_length do
            episode_outputs[ep][step] = outputs[step][ep]
        end
    end
    return 0.5 - pseudoloss(episode_outputs, labels)
end
