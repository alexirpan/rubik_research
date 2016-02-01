-- Implements a Rubik's Cube in Lua
-- TODO move to a different language?

Rubik = {}
UP = 1
LEFT = 2
FRONT = 3
RIGHT = 4
BACK = 5
DOWN = 6


function Rubik:new()
    -- represent cube by stickers on each face
    local newCube = {}
    local grid = {}
    -- faces are numbered 1 to 6
    -- colors are numbered 1 to 6
    -- in the solved state, face number == color number
    for color = 1, 6 do
        grid[color] = {}
        for j = 1, 3 do
            grid[color][j] = {}
            for k = 1,3 do
                grid[color][j][k] = color
            end
        end
    end
    newCube.grid = grid
    self.__index = self
    return setmetatable(newCube, self)
end


function Rubik:solved()
    for color = 1, 6 do
        for j = 1, 3 do
            for k = 1, 3 do
                if self.grid[j][k] ~= color then
                    return false
                end
            end
        end
    end
    return true
end


function rotCW(square)
    local m = table.getn(square)
    local n = table.getn(square[1])

    -- init table
    local new = {}
    for i = 1, n  do
        new[i] = {}
    end
    for i = 1, m do
        for j = 1, n do
            new[j][m-i+1] = square[i][j]
        end
    end
    return new
end


function rotCCW(square)
    local m = table.getn(square)
    local n = table.getn(square[1])

    -- init table
    local new = {}
    for i = 1, n  do
        new[i] = {}
    end
    for i = 1, m do
        for j = 1, n do
            new[n-j+1][i] = square[i][j]
        end
    end
    return new
end


function Rubik:turn()
    return 0
end
