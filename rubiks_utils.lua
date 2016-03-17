--Some assorted utility functions

require 'rubiks'


function dumptable(t)
    -- holy shit Lua doesn't have built in table serialization
    -- THAT'S SO ANNOYING
    local st = ''
    for k, v in pairs(t) do
        print(k, v)
        st = st .. k .. ' ' .. tostring(v) .. '\n'
    end
    return st
end


function sample(dist)
    -- dist is the output from a neural net classifier
    -- In Torch, this is usually the log likelihood, so we need to
    -- exponentiate manually
    -- Uses the O(n) sampling method where n is number of items.
    -- Should be fine
    -- This copies the output to avoid modifing the given distribution
    -- Adding this guarantee is probably worth the small perf loss
    local vals = dist:clone():exp()
    if math.abs(vals:sum() - 1) > 0.001 then
        print('Distribution does not sum to 1!!!!')
        print(vals)
    end
    local rand = torch.uniform() * vals:sum()
    local samp = 1
    while rand > vals[samp] do
        rand = rand - vals[samp]
        samp = samp + 1
    end
    return samp
end


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
        ru:turn(mov)
        if mov <= 6 then
            mov = mov + 6
        else
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
