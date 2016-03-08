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


function generateEpisodes(model, batchSize)
    -- generate a minibatch of batchSize episodes
    local max_length = 50
    local cubes = {}

    for i = 1, batchSize do
        -- A cube is solvable in at most 26 quarter turns
        -- Additionally, any sequence of turns that brings
        -- a cube from solved to solved is of even length
        -- (There's a group theory argument based on parity
        -- of permutations.)
        --
        -- So, scrambles should be both odd and even length, to
        -- make sure the solutions are both odd and even length
        local scramble_length = torch.random(25, 26)
        local cube = _scrambleCube(scramble_length)
        cubes[i] = cube
    end

    -- Tell model to not forget automatically
    model:remember('both')
    model:forget()

    local moves = 0
    local n_finished = 0

    while moves <= max_length and n_finished < batchSize do
        moves = moves + 1
        -- Create a batch from the cubes
        local batch = torch.Tensor(batchSize, N_STICKERS * N_COLORS)
        for i = 1,batchSize do
            batch[i] = cubes[i]:toFeatures():resize(N_STICKERS * N_COLORS)
        end
        -- Wrap in table to pass to sequencer. Then index out the actual output
        local output = model:forward({batch})[1]
        -- now apply moves
        local _, actions = output:max(2)
        print(output)
        print(actions)
        for i = 1,batchSize do
            cubes[i]:turn(actions[i])
        end
    end
end


function computeQvals(episode)
    -- Compute the time discounted Q-values
    -- episode is a table of cubes seen so far
    local epslen = #episode
    local values = torch.Tensor(epslen-1)
    for i = 2,epslen do
        values[i-1] = getReward(episode[i])
    end
    -- have rewards for each action, now propagate back
    for i = epslen-2, 1, -1 do
        values[i] = values[i] + discount * values[i+1]
    end
    return values
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
        generateEpisodes(model, hyperparams.batchSize)
    end
end
