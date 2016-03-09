--[[
Further trains a network by using reinforcement learning
--]]

require 'rnn'
require 'rubiks'
require 'train_rubiks'
require 'test_rubiks'


function getReward(cube)
    if cube:isSolved() then
        return 1
    else
        return 0
    end
end


-- TODO package this into a class?
function initEpisode(start_cube)
    return {
        raw_rewards = {},
        states      = {start_cube:toFeatures():resize(N_STICKERS * N_COLORS)},
        actions     = {},
        finished    = start_cube:isSolved()
    }
end


function updateEpisode(episode, cube, move)
    -- Given an episode, the current state, and the next move, this applies the
    -- given move while adding the required data to episode. If the episode
    -- is already complete, this is a no-op
    --
    -- Returns whether the episode has terminated
    if episode.finished then
        return true
    end
    table.insert(episode.actions, move)
    cube:turn(move)
    table.insert(episode.raw_rewards, getReward(cube))
    table.insert(episode.states, cube:toFeatures():resize(N_STICKERS * N_COLORS))
    episode.finished = cube:isSolved()
    return episode.finished
end


function generateEpisodes(model, batchSize)
    -- generate a minibatch of batchSize episodes
    local max_length = 50
    local cubes = {}
    local episodes = {}

    for i = 1, batchSize do
        -- A cube is solvable in at most 26 quarter turns
        -- Additionally, any sequence of turns that brings
        -- a cube from solved to solved is of even length
        -- (There's a group theory argument based on parity
        -- of permutations.)
        --
        -- So, scrambles should be both odd and even length, to
        -- make sure the solutions are both odd and even length
        local scramble_length = torch.random(2,3)
        local cube = _scrambleCube(scramble_length)
        cubes[i] = cube
        episodes[i] = initEpisode(cube)
    end

    -- Tell model to not forget automatically
    model:remember('both')
    model:forget()

    local moves = 0
    local n_finished = 0

    while moves <= max_length and n_finished < batchSize do
        n_finished = 0
        moves = moves + 1
        -- Create a batch from the cubes
        local batch = torch.Tensor(batchSize, N_STICKERS * N_COLORS)
        for i = 1,batchSize do
            batch[i] = cubes[i]:toFeatures():resize(N_STICKERS * N_COLORS)
        end
        -- Wrap in table to pass to sequencer. Then index out the actual output
        local output = model:forward({batch})[1]
        -- TODO make this sample from output distribution instead of taking max
        -- now apply move for everything in batch
        local _, actions = output:max(2)
        for i = 1,batchSize do
            if updateEpisode(episodes[i], cubes[i], actions[i][1]) then
                n_finished = n_finished + 1
            end
        end
    end
    return episodes
end


function computeDiscountedRewards(episode, discount)
    -- Compute the time discounted reward
    --
    -- TODO decide if these should be bashed into Tensors or not
    local n_vals = #episode.actions
    -- value for (prev_state, action)
    -- state + action is index i (next state is index i+1)
    local rewards = {}
    for i = 1, n_vals do
        rewards[i] = episode.raw_rewards[i]
    end
    -- propagate back
    -- (this uses the biased reward for policy gradient)
    for i = n_vals - 1, 1, -1 do
        rewards[i] = rewards[i] + discount * rewards[i+1]
    end
    episode.rewards = rewards
end


local from_cmd_line = (debug.getinfo(3).name == nil)

if from_cmd_line then
    local NOMODEL = '.'
    local NOFILE = '.'
    local SCRAMBLE_DEFAULT = -1
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text("RL script for Rubik's Cube neural net solve")
    cmd:text("The default behavior is to load the model, then")
    cmd:text("run some RL episodes to fine-tune it.")
    cmd:option('--model', NOMODEL, 'File path to the model')
    cmd:option('--savefile', NOFILE, 'Where to save the refined model')
    cmd:option('--batchsize', 8, 'Batchsize (# episodes run before update')
    cmd:option('--nbatches', 1, 'Number of batches to use')
    cmd:text()

    opt = cmd:parse(arg or {})

    if opt.model == NOMODEL then
        print('No model path specified, exiting')
        return
    end
    if opt.savefile == NOMODEL then
        print('Save file for refined model is required')
        return
    end

    hyperparams = {
        seed = 123,
        learningRate = 0.1,
        batchSize = opt.batchsize,
        nBatches = opt.nbatches,
        discount = 0.95
    }
    torch.manualSeed(hyperparams.seed)

    data = torch.load(opt.model, 'ascii')
    model = data.model
    n_trials = opt.ntest

    infostring = string.format(
        'Running RL on %s with these parameters:\n' ..
        'Batch size %d, n_batches %d, TODO',
        opt.model,
        opt.batchsize,
        opt.nbatches
    )

    print(infostring)
    for i = 1, hyperparams.nBatches do
        eps = generateEpisodes(model, hyperparams.batchSize)
        for j = 1, hyperparams.batchSize do
            computeDiscountedRewards(eps[j], hyperparams.discount)
        end
        print(eps)
    end
end
