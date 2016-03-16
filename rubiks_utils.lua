--Some assorted utility functions

require 'rubiks'


function scrambleCube(length)
    -- Generates a random scramble of a Rubik's Cube
    -- Returns two items, an Rubik's cube object and
    -- a table of the moves needed to reverse the
    -- scramble
    local ru = Rubik:new()
    local moves = {}
    for j = 1, length do
        local mov = torch.random(1, N_MOVES)
        if mov <= 6 then
            ru:turnCW(mov)
            mov = mov + 6
        else
            ru:turnCCW(mov - 6)
            mov = mov - 6
        end
        moves[length + 1 - j] = mov
    end
    return ru, moves
end
