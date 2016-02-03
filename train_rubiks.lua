require 'rnn'
require 'rubiks'

-- Number of moves to use when scrambling
EPISODE_LENGTH = 3
-- Number of possible turns of the cube
N_MOVES = 12


function _generateEpisodes(n_episodes)
    local eps = torch.Tensor(n_episodes * EPISODE_LENGTH, N_STICKERS, N_COLORS):zero()
    local eps_labels = torch.Tensor(n_episodes * EPISODE_LENGTH):zero()

    for i = 1, n_episodes do
        local ru = Rubik:new()
        local moves = {}
        for j = 1, EPISODE_LENGTH do
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
            eps[(i-1)*EPISODE_LENGTH + j] = ru:toFeatures()
            eps_labels[(i-1)*EPISODE_LENGTH + j] = mov
        end
    end

    return eps, eps_labels
end


function createDataset(n_train, n_valid, n_test)
    -- generates a triple of 3 datasets: training, validation, and test
    --
    -- n_train: the number of training episodes
    -- n_valid: the number of validation episodes
    -- n_test: the number of test episodes
    local train, train_lab = _generateEpisodes(n_train)
    local valid, valid_lab = _generateEpisodes(n_valid)
    local test, test_lab = _generateEpisodes(n_test)
    return {
        train = train,
        train_labels = train_lab,
        valid = valid,
        valid_labels = valid_lab,
        test = test,
        test_labels = test_lab
    }
end


hiddenSize = 100


function fullyConnected()
    -- return a fully connected model with 1 hidden layer
    local fcnn = nn.Sequential()
    fcnn:add( nn.Linear(N_STICKERS * N_COLORS, hiddenSize) )
    fcnn:add( nn.Tanh() )
    fcnn:add( nn.Linear(hiddenSize, N_MOVES) )
    fcnn:add( nn.LogSoftMax() )

    local loss = nn.ClassNLLCriterion()

    return fcnn, loss
end


-- training
torch.manualSeed(987)
epoch = 1
n_epochs = 20
batchSize = 8
learningRate = 0.1
n_train = 10000
n_valid = 1000
n_test = 5000


data = createDataset(n_train, n_valid, n_test)
-- flatten for fully connected
data['train']:resize(n_train * EPISODE_LENGTH,
                     N_STICKERS * N_COLORS)
data['valid']:resize(n_test * EPISODE_LENGTH,
                     N_STICKERS * N_COLORS)
data['test']:resize(n_test * EPISODE_LENGTH,
                     N_STICKERS * N_COLORS)
model, loss = fullyConnected()

while epoch < n_epochs do
    print('Starting epoch', epoch)
    local err, correct = 0, 0

    for ind = 1, n_train / batchSize do
        local inputs, targets = {}, {}
        start = (ind - 1) * batchSize + 1
        for i = 1, batchSize do
            inputs[i] = data['train'][start + i-1]
            targets[i] = data['train_labels'][start + i-1]
        end

        local outputs = {}
        local gradOutputs, gradInputs = {}, {}

        for i = 1, batchSize do
            -- forward sequence
            model:zeroGradParameters()
            outputs[i] = model:forward(inputs[i])
            err = err + loss:forward(outputs[i], targets[i])

            -- compute accuracy
            _, best = outputs[i]:max(1)
            if best[1] == targets[i] then
                correct = correct + 1
            end

            -- backprop
            gradOutputs[i] = loss:backward(outputs[i], targets[i])
            model:backward(inputs[i], gradOutputs[i])
            model:updateParameters(learningRate)
        end
    end
    print(string.format("Epoch %d: Average training loss = %f, training accuracy = %f %%", epoch, err / n_train, correct / n_train * 100))
    epoch = epoch + 1

    -- test error
    err, correct = 0, 0

    for i = 1, n_test do
        local inputs, targets = {}, {}
        local outputs = {}
        inputs[i] = data['test'][i]
        targets[i] = data['test_labels'][i]
        outputs[i] = model:forward(inputs[i])
        err = err + loss:forward(outputs[i], targets[i])
        _, best = outputs[i]:max(1)
        if best[1] == targets[i] then
            correct = correct + 1
        end
    end

    print(string.format("Test loss = %f, test accuracy = %f %%", err / n_test, correct / n_test * 100))
end
