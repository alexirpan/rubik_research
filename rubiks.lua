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


function Rubik:neighborFaces(face)
    -- Returns the neighboring sides of the given face in the order that
    -- makes a CW turn easy. (In the returned list, stickers on the first face
    -- go to the second face, second face to third face, and so on.)
    --
    -- For a CCW turn, can just use the reverse
    --
    -- Table lookup should be faster than if statement tower
    -- ULFRBD order
    local answers= {
        { LEFT, BACK, RIGHT, FRONT },
        { UP, FRONT, DOWN, BACK },
        { UP, RIGHT, DOWN, LEFT },
        { UP, BACK, DOWN, FRONT },
        { UP, LEFT, DOWN, RIGHT },
        { LEFT, FRONT, RIGHT, BACK }
    }
    return answers[face]
end


function Rubik:stickerMove(face, neigh)
    -- Helper method for moving stickers not on the face
    -- indices == where the stickers should be moving to on that face
    local indices = {}
    if face == UP then
        indices[LEFT]  = {{1,1},{1,2},{1,3}}
        indices[BACK]  = {{1,1},{1,2},{1,3}}
        indices[RIGHT] = {{1,1},{1,2},{1,3}}
        indices[FRONT] = {{1,1},{1,2},{1,3}}
    elseif face == LEFT then
        indices[UP]    = {{1,1},{2,1},{3,1}}
        indices[FRONT] = {{1,1},{2,1},{3,1}}
        indices[DOWN]  = {{1,1},{2,1},{3,1}}
        indices[BACK]  = {{3,3},{2,3},{1,3}}
    elseif face == FRONT then
        indices[UP]    = {{3,1},{3,2},{3,3}}
        indices[RIGHT] = {{1,1},{2,1},{3,1}}
        indices[DOWN]  = {{1,3},{1,2},{1,1}}
        indices[LEFT]  = {{3,3},{2,3},{1,3}}
    elseif face == RIGHT then
        indices[UP]    = {{3,3},{2,3},{1,3}}
        indices[BACK]  = {{1,1},{2,1},{3,1}}
        indices[DOWN]  = {{3,3},{2,3},{1,3}}
        indices[FRONT] = {{3,3},{2,3},{1,3}}
    elseif face == BACK then
        indices[UP]    = {{1,3},{1,2},{1,1}}
        indices[LEFT]  = {{1,1},{2,1},{3,1}}
        indices[DOWN]  = {{3,1},{3,2},{3,3}}
        indices[RIGHT] = {{3,3},{2,3},{1,3}}
    elseif face == DOWN then
        indices[LEFT]  = {{3,1},{3,2},{3,3}}
        indices[FRONT] = {{3,1},{3,2},{3,3}}
        indices[RIGHT] = {{3,1},{3,2},{3,3}}
        indices[BACK]  = {{3,1},{3,2},{3,3}}
    end

    -- extract stickers from indices
    local stickers = {}
    for face, inds in pairs(indices) do
        stickers[face] = {}
        for i = 1, 3 do
            stickers[face][i] = self.grid[face][inds[i][1]][inds[i][2]]
        end
    end

    -- move them into the right spot
    local replaceWith = {}
    replaceWith[neigh[2]] = neigh[1]
    replaceWith[neigh[3]] = neigh[2]
    replaceWith[neigh[4]] = neigh[3]
    replaceWith[neigh[1]] = neigh[4]
    for _, face in ipairs(neigh) do
        local inds = indices[face]
        for i = 1, 3 do
            self.grid[face][inds[i][1]][inds[i][2]] = stickers[replaceWith[face]][i]
        end
    end
end


function Rubik:turnCW(face)
    -- turn the specified face 90 degrees clockwise
    -- Follows standard move convention. Ex: turnCW(LEFT) is an L move and
    -- turnCW(RIGHT) is an R move
    self.grid[face] = rotCW(self.grid[face])
    self:stickerMove(face, self:neighborFaces(face))
end


function Rubik:turnCCW(face)
    -- turn the specified face 90 degrees counterclockwise
    -- Follows standard move convention. Ex: turnCCW(LEFT) is an L' move and
    -- turnCCW(RIGHT) is an R' move
    self.grid[face] = rotCCW(self.grid[face])
    local neigh = self:neighborFaces(face)
    neigh[1], neigh[4] = neigh[4], neigh[1]
    neigh[2], neigh[3] = neigh[3], neigh[2]
    self:stickerMove(face, neigh)
end


function rubikTurnTests()
    local ru = Rubik:new()
    -- for testing, initialize stickers uniquely
    local lab = 1
    for i = 1, 6 do
        for j = 1, 3 do
            for k = 1, 3 do
                ru.grid[i][j][k] = lab
                lab = lab + 1
            end
        end
    end
end
