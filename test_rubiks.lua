require 'rnn'
require 'rubiks'


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
        local output = model:forward(start_cube:toFeatures())
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


