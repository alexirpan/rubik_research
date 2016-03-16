--Some assorted utility functions

require 'rubiks'


function randomCubeEpisode(length)
    -- Generates a random scramble of a Rubik's Cube
    -- Returns two items:
    --
    -- - A Torch Tensor of all features seen in the episode
    -- - A Torch Tensor of the moves needed to solve the cube
    local ru = Rubik:new()
    local episode = torch.Tensor(length, N_STICKERS, N_COLORS):zero()
    local moves = torch.LongTensor(length):zero()

    -- Store the generated episodes in backwards order.
    -- When passing a sequence to the RNN, we want to end at the solved
    -- state, not start from it.
    for j = length, 1, -1 do
        local mov = torch.random(1, N_MOVES)
        -- the correct label is the inverse of the move, after applying
        -- move modify appropriately
        if mov <= 6 then
            ru:turnCW(mov)
            mov = mov + 6
        else
            ru:turnCCW(mov - 6)
            mov = mov - 6
        end
        episode[j] = ru:toFeatures()
        moves[j] = mov
    end

    return episode, moves
end


function randomCube(length)
    -- Generates a random scramble, returning just the Rubik's
    -- Cube object. Creating features at every timestep is a huge
    -- timesink for the RL portion
    local ru = Rubik:new()
    for j = 1, length do
        local mov = torch.random(1, N_MOVES)
        -- the correct label is the inverse of the move, after applying
        -- move modify appropriately
        if mov <= 6 then
            ru:turnCW(mov)
        else
            ru:turnCCW(mov - 6)
        end
    end
    return ru
end
