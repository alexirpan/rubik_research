require 'rnn'
require 'rubiks'

-- Number of moves to use when scrambling
EPISODE_LENGTH = 2
-- Number of possible turns of the cube
N_MOVES = 12


function _scrambleCube(length)
    -- TODO better refactor this with the method below to avoid code duplication
    local ru = Rubik:new()
    for j = 1, length do
        local mov = torch.random(1, N_MOVES)
        if mov <= 6 then
            ru:turnCW(mov)
        else
            ru:turnCCW(mov - 6)
        end
    end
    return ru
end

function _generateEpisodes(n_episodes)
    local eps = torch.Tensor(n_episodes * EPISODE_LENGTH, N_STICKERS, N_COLORS):zero()
    -- using a LongTensor makes later comparison easier. These are class indices so it's fine
    local eps_labels = torch.Tensor(n_episodes * EPISODE_LENGTH):zero():type('torch.LongTensor')

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


rho = 10


function plainRecurrent()
    -- return a plain recurrent net
    -- first create recurrent module
    local recur = nn.Recurrent(
        hiddenSize,                                    -- output size
        nn.Linear(N_STICKERS * N_COLORS, hiddenSize),  -- input (output of this must match size above)
        nn.Linear(hiddenSize, hiddenSize),             -- feedbacks previous output tensor to transfer
        nn.Tanh(),                                     -- transfer, nonlinearity applied to sum of input and feedback
        rho                                            -- backprop steps to go back in time
    )
    -- then add in the final classification layer
    local rnn = nn:Sequential()
    rnn:add(recur)
    rnn:add(nn.Linear(hiddenSize, N_MOVES))
    rnn:add(nn.LogSoftMax())
    -- wrap with recursor (TODO needed?)
    rnn = nn.Recursor(rnn, rho)

    local loss = nn.ClassNLLCriterion()

    return rnn, loss
end


function trainFullModel()
    -- training
    torch.manualSeed(987)
    epoch = 1
    n_epochs = 40
    batchSize = 8
    learningRate = 0.1
    n_train = 50000
    n_valid = 1000  -- VALIDATION SET NOT USED
    n_test = 10000


    data = createDataset(n_train, n_valid, n_test)
    -- flatten for fully connected
    data['train']:resize(n_train * EPISODE_LENGTH,
                         N_STICKERS * N_COLORS)
    data['valid']:resize(n_valid * EPISODE_LENGTH,
                         N_STICKERS * N_COLORS)
    data['test']:resize(n_test * EPISODE_LENGTH,
                         N_STICKERS * N_COLORS)
    model, loss = fullyConnected()

    -- for the fully connected model we don't have a notion of sequences
    -- So there are actually n_train * EPISODE_LENGTH samples total
    n_train = n_train * EPISODE_LENGTH
    n_valid = n_valid * EPISODE_LENGTH
    n_test = n_test * EPISODE_LENGTH

    best_acc = 0

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

        -- test error
        local test_err, test_correct = 0, 0

        for i = 1, n_test do
            local inputs, targets = {}, {}
            local outputs = {}
            inputs[i] = data['test'][i]
            targets[i] = data['test_labels'][i]
            outputs[i] = model:forward(inputs[i])
            test_err = test_err + loss:forward(outputs[i], targets[i])
            _, best = outputs[i]:max(1)
            if best[1] == targets[i] then
                test_correct = test_correct + 1
            end
        end
        print(string.format("Test loss = %f, test accuracy = %f %%", test_err / n_test, test_correct / n_test * 100))

        -- save epoch data
        saved = {
            model = model,
            train_err = err / n_train,
            train_acc = correct / n_train * 100,
            test_err = test_err / n_test,
            test_acc = test_correct / n_test * 100
        }
        if saved.test_acc > best_acc then
            best_acc = saved.test_acc
            filename = 'models/rubiks_best'
            torch.save(filename, saved)
        end

        filename = 'models/rubiks_epoch' .. epoch
        torch.save(filename, saved)

        epoch = epoch + 1
    end
end


-- (TODO) MY GOD REFACTOR THIS DON'T JUST COPY PASTE
-- (TODO) Write this such that I can pass the entire sequence in one :forward call
-- and get the gradient in one :backward call. Doing it once explicitly to get
-- a better intuition
function trainPlainRecurModel()
    -- training
    torch.manualSeed(987)
    epoch = 1
    n_epochs = 40
    batchSize = 8
    learningRate = 0.1
    n_train = 50000
    n_valid = 1000
    n_test = 10000


    data = createDataset(n_train, n_valid, n_test)
    -- flatten last two axes
    data['train']:resize(n_train * EPISODE_LENGTH,
                         N_STICKERS * N_COLORS)
    data['valid']:resize(n_test * EPISODE_LENGTH,
                         N_STICKERS * N_COLORS)
    data['test']:resize(n_test * EPISODE_LENGTH,
                         N_STICKERS * N_COLORS)
    train = data['train']
    train_labels = data['train_labels']
    valid = data['valid']
    valid_labels = data['valid_labels']
    test = data['test']
    test_labels = data['test_labels']

    model, loss = plainRecurrent()

    best_acc = 0

    while epoch < n_epochs do
        print('Starting epoch', epoch)
        local err, correct = 0, 0

        -- n_train is the number of sequences
        -- we do (batchSize) sequences at once
        for ind = 1, n_train / batchSize do
            -- generate a batch of sequences
            -- as constructed, each EPISODE_LENGTH block is a new sequence
            -- So to get batchSize sequences at once, we need to start pulling from
            -- (start, start + EPISODE, start + 2*EPISODE, ...)
            start = (ind - 1) * batchSize * EPISODE_LENGTH + 1
            local seqIndices = torch.range(
                start, start + (batchSize - 1) * EPISODE_LENGTH, EPISODE_LENGTH
            ):type('torch.LongTensor')

            local inputs, targets = {}, {}
            for step = 1, EPISODE_LENGTH do
                inputs[step] = train:index(1, seqIndices)
                targets[step] = train_labels:index(1, seqIndices)
                seqIndices:add(1)
            end

            -- run sequence forward through rnn
            model:zeroGradParameters()
            model:forget()  -- forget past time steps

            local outputs = {}
            -- reset seqIndices to check accuracy
            seqIndices = torch.range(
                start, start + (batchSize - 1) * EPISODE_LENGTH, EPISODE_LENGTH
            ):type('torch.LongTensor')

            for step = 1, EPISODE_LENGTH do
                outputs[step] = model:forward(inputs[step])
                err = err + loss:forward(outputs[step], targets[step])
                -- compute accuracy
                -- shape of output at each step is (batchSize, N_MOVES)
                _, predicted = outputs[step]:max(2)
                predicted:resize(batchSize)
                trueVal = targets[step]
                correct = correct + predicted:eq(trueVal):sum()
            end

            -- backward sequence (backprop through time)
            local gradOutputs, gradInputs = {}, {}
            for step = EPISODE_LENGTH, 1, -1 do
                gradOutputs[step] = loss:backward(outputs[step], targets[step])
                gradInputs[step] = model:backward(inputs[step], gradOutputs[step])
            end

            -- and update
            model:updateParameters(learningRate)
        end
        -- for averages, we need to account for there being n_episodes * EPISODE_LENGTH samples total
        err = err / (n_train * EPISODE_LENGTH)
        correct = correct / (n_train * EPISODE_LENGTH) * 100
        print(string.format(
            "Epoch %d: Average training loss = %f, training accuracy = %f %%",
            epoch,
            err,
            correct
        ))

        -- test error
        local test_err, test_correct = 0, 0

        for i = 1, n_test do
            for step = 1, EPISODE_LENGTH do
                model:forget()
                input = test[(i-1) * EPISODE_LENGTH + step]
                target = test_labels[(i-1) * EPISODE_LENGTH + step]
                output = model:forward(input)
                test_err = test_err + loss:forward(output, target)
                _, best = output:max(1)
                if best[1] == target then
                    test_correct = test_correct + 1
                end
            end
        end
        -- again account for episode length
        test_err = test_err / (n_test * EPISODE_LENGTH)
        test_correct = test_correct / (n_test * EPISODE_LENGTH) * 100
        print(string.format("Test loss = %f, test accuracy = %f %%", test_err, test_correct))

        -- save epoch data
        saved = {
            model = model,
            train_err = err,
            train_acc = correct,
            test_err = test_err,
            test_acc = test_correct
        }
        if saved.test_acc > best_acc then
            best_acc = saved.test_acc
            filename = 'models/rubiks_best'
            torch.save(filename, saved)
        end

        filename = 'models/rubiks_epoch' .. epoch
        torch.save(filename, saved)

        epoch = epoch + 1
    end
end
