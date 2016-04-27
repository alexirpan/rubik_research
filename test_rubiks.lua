require 'rnn'
require 'rubiks'
require 'rubiks_utils'

require 'averager'


function trySolving(model, start_cube, randomized)
    -- applies moves to the cube until it hits the solved state, or
    -- until it fails too many times
    --
    -- returns (whether cube was solved, moves_taken)
    --
    -- at most 27 turns are needed. The fastest human methods use around
    -- 40 to 60 turns
    local max_length = 50

    local moves = 0
    local moves_done = {}

    -- The model is wrapped in a Sequencer
    -- We need to tell it not to forget automatically
    model:remember('both')
    model:forget()
    while moves <= max_length and not start_cube:isSolved() do
        -- Sequencer expects a table of T timesteps
        -- So give it a table of 1 timestep
        local feat = start_cube:toFeatures():resize(N_STICKERS * N_COLORS)
        if CUDA then
            feat = feat:cuda()
        end
        local output = model:forward(
            {feat}
        )[1]
        local _, action = output:max(1)
        action = action[1]

        start_cube:turn(action)
        table.insert(moves_done, action)
        moves = moves + 1
    end

    return start_cube:isSolved(), moves_done
end


function solveCubes(n_trials, randomized)
    local solved_count = 0
    local solved_hist = {}
    local solved_length = 0

    for i = 1, n_trials do
        if i % 100 == 0 then
            print(n_trials - i, 'steps left')
        end
        cube = randomCube(scramble_length)
        solved, moves = trySolving(model, cube, randomized)
        if solved then
            local sol_len = table.getn(moves)

            solved_count = solved_count + 1
            solved_length = solved_length + sol_len
            if solved_hist[sol_len] == nil then
                solved_hist[sol_len] = 1
            else
                solved_hist[sol_len] = solved_hist[sol_len] + 1
            end

        end
    end
    return solved_count, solved_hist, solved_length
end


-- TODO make this also accept GPU
local from_cmd_line = (debug.getinfo(3).name == nil)

if from_cmd_line then
    local NOMODEL = 'NOMODEL'
    local NOFILE = 'NOFILE'
    local SCRAMBLE_DEFAULT = -1
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text("Testing script for Rubik's Cube neural net solve")
    cmd:text("The default behavior is to load the model, look up")
    cmd:text("the episode length from the hyperparams, then test scrambles")
    cmd:text("of length 1 longer.")
    cmd:option('--model', NOMODEL, 'File path to the model')
    cmd:option('--savefile', NOFILE, 'Where to save solve data')
    cmd:option('--ntest', 10000, 'Number of cubes to test on')
    cmd:option('--scramblelen', SCRAMBLE_DEFAULT, 'Length of scramble to use (default episode length + 1')
    cmd:option('--gpu', 0, 'Use GPU or not')
    cmd:option('--randomized', 0, 'Use argmax policy or randomized policy. Not implemented yet.')
    cmd:text()

    opt = cmd:parse(arg or {})
    torch.manualSeed(54321)

    CUDA = (opt.gpu ~= 0)
    if CUDA then
        require 'cutorch'
        require 'cunn'
        cutorch.manualSeedAll(54321)
    end

    if opt.model == NOMODEL then
        print('No model path specified, exiting')
        return
    end

    if opt.savefile ~= NOFILE then
        savefile = io.open(opt.savefile, 'w')
    end

    data = torch.load(opt.model, 'ascii')
    model = data.model
    if CUDA then
        model = model:cuda()
    end
    episode_length = data.hyperparams.episode_length
    params_string = dumptable(data.hyperparams)
    if opt.scramblelen == SCRAMBLE_DEFAULT then
        scramble_length = episode_length + 1
    else
        scramble_length = opt.scramblelen
    end
    n_trials = opt.ntest

    infostring = string.format(
        'Testing model with these parameters:\n' ..
        '%s\n' ..
        'Episode length %d, test_acc %f %%. Using scrambles of length %d',
        params_string,
        episode_length,
        data.test_acc,
        scramble_length
    )

    print(infostring)
    if savefile ~= nil then
        savefile:write(infostring .. '\n')
    end

    solved_count, solved_hist, solved_length = solveCubes(n_trials, (opt.randomized ~= 0))
    if solved_count == 0 then
        infostring = 'Solved no cubes :('
    else
        infostring = string.format(
            'Solved %.2f%% of cubes, average solve length %f\n' ..
            'Distribution of solve lengths on success\n' ..
            '%s',
            solved_count / n_trials * 100,
            solved_length / solved_count,
            dumptable(solved_hist)
        )
    end

    print(infostring)
    if savefile ~= nil then
        savefile:write(infostring .. '\n')
        savefile:close()
    end
end
