require 'rnn'
require 'rubiks'
require 'train_rubiks'


function trySolving(model, start_cube)
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

    while moves <= max_length and not start_cube:isSolved() do
        local output = model:forward(
            start_cube:toFeatures():resize(N_STICKERS * N_COLORS)
        )
        local _, action = output:max(1)
        action = action[1]

        if action <= 6 then
            start_cube:turnCW(action)
        else
            start_cube:turnCCW(action - 6)
        end
        table.insert(moves_done, action)
        moves = moves + 1
    end

    return start_cube:isSolved(), moves_done
end


data = torch.load('models/rubiks_best')
model = data.model

-- EPISODE_LENGTH comes from requiring train_rubiks

print(string.format(
    'Testing model, trained with episode length %d, test acc %f',
    EPISODE_LENGTH, data.test_acc
))


n_trials = 10000

solved_count = 0
solved_hist = {}
solved_length = 0

for i = 1, n_trials do
    if i % 100 == 0 then
        print(n_trials - i, 'steps left')
    end
    cube = _scrambleCube(EPISODE_LENGTH+3)
    solved, moves = trySolving(model, cube)
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

if solved_count == 0 then
    print('Solved no cubes :(')
else
    print(string.format(
        'Solved %.2f%% of cubes, average solve length %f',
        solved_count / n_trials * 100, solved_length / solved_count
    ))
    print(solved_hist)
end
