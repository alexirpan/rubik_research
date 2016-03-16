require 'rnn'
require 'rubiks'
require 'rubiks_utils'


function _setupHyperparams()
    -- The old version of this code used the hyperparams as global variables
    -- It's ugly, but for now we expose all of these globally
    -- (Note hyperparams is not instantiated unless this is run from the command
    -- line. If it is, then it is initialized based on command line arguments)
    torch.manualSeed(hyperparams.seed)
    if CUDA then
        cutorch.manualSeedAll(hyperparams.seed)
    end
    n_epochs = hyperparams.n_epochs
    batchSize = hyperparams.batchSize
    learningRate = hyperparams.learningRate
    n_train = hyperparams.n_train
    n_valid = hyperparams.n_valid
    n_test = hyperparams.n_test
    EPISODE_LENGTH = hyperparams.episode_length
    saveTo = hyperparams.saved_to
    hiddenSize = hyperparams.hiddenSize
    rho = hyperparams.rho
end


function generateEpisodes(n_episodes)
    local eps = torch.Tensor(n_episodes * EPISODE_LENGTH, N_STICKERS, N_COLORS):zero()
    -- using a LongTensor makes later comparison easier. These are class indices so it's fine
    local eps_labels = torch.LongTensor(n_episodes * EPISODE_LENGTH):zero()

    for i = 1, n_episodes do
        local episode, moves = randomCubeEpisode(EPISODE_LENGTH)
        local start = (i-1) * EPISODE_LENGTH
        eps[{ {start+1, start+EPISODE_LENGTH} }] = episode
        eps_labels[{ {start+1, start+EPISODE_LENGTH} }] = moves
    end

    if CUDA then
        -- TODO labels should still be a Long Tensor right? or is cuda
        -- better?
        eps = eps:cuda()
        eps_labels = eps_labels:cuda()
    end

    return eps, eps_labels
end


function _createData(n_train, n_valid, n_test, in_parallel)
    -- generates a triple of 3 datasets: training, validation, and test
    --
    -- n_train: the number of training episodes
    -- n_valid: the number of validation episodes
    -- n_test: the number of test episodes
    local train, train_lab = generateEpisodes(n_train)
    local valid, valid_lab = generateEpisodes(n_valid)
    local test, test_lab = generateEpisodes(n_test)

    local datasets = {
        train = train,
        train_labels = train_lab,
        valid = valid,
        valid_labels = valid_lab,
        test = test,
        test_labels = test_lab
    }

    if not in_parallel then
        return datasets
    else
        parallel.parent:send(datasets)
        parallel.yield()
    end
end


function createDataset(n_train, n_valid, n_test)
    return _createData(n_train, n_valid, n_test, false)
end


function createDatasetInParallel(n_train, n_valid, n_test)
    require 'parallel'
    -- ALWAYS, ALWAYS wrap this in a pcall to make sure
    -- children are cleaned up
    -- (so far this is unused. May add it back in if needed
    local code = function()
        _createData(n_train, n_valid, n_test, true)
    end
    child = parallel.fork()
    child:exec(code)

    local datasets = child:receive()
    child:join()
    return datasets
end


function fullyConnected()
    -- return a fully connected model with 1 hidden layer
    -- This is wrapped in a Sequencer to allow using one
    -- training method for all the different models. All the
    -- sequencer does is automate a few things when running
    -- on multiple sequences, and requiring 'rnn' makes every
    -- model have these methods defined. So, this shouldn't
    -- change the functionality in any way
    local fcnn = nn.Sequential()
    fcnn:add( nn.Linear(N_STICKERS * N_COLORS, hiddenSize) )
    fcnn:add( nn.Tanh() )
    fcnn:add( nn.Linear(hiddenSize, N_MOVES) )
    fcnn:add( nn.LogSoftMax() )

    fcnn = nn.Sequencer(fcnn)

    local loss = nn.SequencerCriterion(nn.ClassNLLCriterion())

    return fcnn, loss
end


function biggerFullyConnected()
    -- A 2 layer hidden net with # of parameters similar to a
    -- 100 hidden size LSTM
    local fcnn = nn.Sequential()
    fcnn:add( nn.Linear(N_STICKERS * N_COLORS, 200) )
    fcnn:add( nn.Tanh() )
    fcnn:add( nn.Linear(200, 150) )
    fcnn:add( nn.Tanh() )
    fcnn:add( nn.Linear(150, N_MOVES) )
    fcnn:add( nn.LogSoftMax() )

    fcnn = nn.Sequencer(fcnn)

    local loss = nn.SequencerCriterion(nn.ClassNLLCriterion())

    return fcnn, loss
end


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

    -- wrap with sequencer
    rnn = nn.Sequencer(rnn)

    -- use sequential loss
    local loss = nn.SequencerCriterion(nn.ClassNLLCriterion())

    return rnn, loss
end


function LSTM()
    -- a 1 layer LSTM
    local lstm = nn:Sequential()
    lstm:add(
        nn.LSTM(
            N_STICKERS * N_COLORS,
            hiddenSize,
            rho
        )
    )
    lstm:add(nn.Linear(hiddenSize, N_MOVES))
    lstm:add(nn.LogSoftMax())

    -- wrap with sequencer
    lstm = nn.Sequencer(lstm)
    local loss = nn.SequencerCriterion(nn.ClassNLLCriterion())

    return lstm, loss
end


function trainModel(model, loss)
    local timer = torch.Timer()
    print('Creating data')
    data = createDataset(n_train, n_valid, n_test)
    -- flatten last two axes
    data['train']:resize(n_train * EPISODE_LENGTH,
                         N_STICKERS * N_COLORS)
    data['valid']:resize(n_test * EPISODE_LENGTH,
                         N_STICKERS * N_COLORS)
    data['test']:resize(n_test * EPISODE_LENGTH,
                         N_STICKERS * N_COLORS)

    seconds = timer:time().real
    minutes = math.floor(seconds / 60)
    seconds = seconds - 60 * minutes
    print(string.format('Spent %d minutes %f seconds creating data', minutes, seconds))

    train = data['train']
    train_labels = data['train_labels']
    valid = data['valid']
    valid_labels = data['valid_labels']
    test = data['test']
    test_labels = data['test_labels']

    best_acc = 0
    epoch = 1
    while epoch <= n_epochs do
        print('Starting epoch', epoch)
        local err, correct = 0, 0

        -- go through batches in random order
        local nBatches = n_train / batchSize
        local perm = torch.randperm(nBatches)

        -- n_train is the number of sequences
        -- we do (batchSize) sequences at once
        for j = 1, nBatches do
            ind = perm[j]
            -- generate a batch of sequences
            -- as constructed, each EPISODE_LENGTH block is a new sequence
            -- So to get batchSize sequences at once, we need to start pulling from
            -- (start, start + EPISODE, start + 2*EPISODE, ...)
            start = (ind - 1) * batchSize * EPISODE_LENGTH + 1
            local seqIndices = torch.LongTensor():range(
                start, start + (batchSize - 1) * EPISODE_LENGTH, EPISODE_LENGTH
            )

            local inputs, targets = {}, {}
            for step = 1, EPISODE_LENGTH do
                inputs[step] = train:index(1, seqIndices)
                targets[step] = train_labels:index(1, seqIndices)
                seqIndices:add(1)
            end

            -- run sequence forward through rnn
            model:zeroGradParameters()
            model:forget()  -- forget past time steps

            local outputs = model:forward(inputs)
            err = err + loss:forward(outputs, targets)

            -- reset seqIndices to check accuracy
            seqIndices = torch.LongTensor():range(
                start, start + (batchSize - 1) * EPISODE_LENGTH, EPISODE_LENGTH
            )

            for step = 1, EPISODE_LENGTH do
                -- compute accuracy
                -- shape of output at each step is (batchSize, N_MOVES)
                _, predicted = outputs[step]:max(2)
                predicted:resize(batchSize)
                trueVal = targets[step]
                correct = correct + predicted:eq(trueVal):sum()
            end

            -- backward sequence (backprop through time)
            local gradOutputs = loss:backward(outputs, targets)
            local gradInputs = model:backward(inputs, gradOutputs)

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
            local start = (i-1) * EPISODE_LENGTH + 1
            input_ = test:narrow(1, start, EPISODE_LENGTH)
            -- The sequencer interface expects a Lua table of
            -- seqlen entries, each of which is a batchsize x featsize tensor
            input = {}
            for step = 1,EPISODE_LENGTH do
                input[step] = input_[step]
            end
            output = model:forward(input)
            target_ = test_labels:narrow(1, start, EPISODE_LENGTH)
            -- Again, table instead of tensor
            target = {}
            for step = 1,EPISODE_LENGTH do
                target[step] = target_[step]
            end
            test_err = test_err + loss:forward(output, target)

            for step = 1, EPISODE_LENGTH do
                _, best = output[step]:max(1)
                if best[1] == target[step] then
                    test_correct = test_correct + 1
                end
            end
        end
        -- again account for episode length
        test_err = test_err / (n_test * EPISODE_LENGTH)
        test_correct = test_correct / (n_test * EPISODE_LENGTH) * 100
        print(string.format("Test loss = %f, test accuracy = %f %%", test_err, test_correct))

        -- save epoch data
        if CUDA then
            -- convert the model back to DoubleTensor before saving
            -- Otherwise it's only loadable on machines with GPU
            model = model:double()
        end

        saved = {
            model = model,
            train_err = err,
            train_acc = correct,
            test_err = test_err,
            test_acc = test_correct,
            hyperparams = hyperparams
        }
        if saved.test_acc > best_acc then
            best_acc = saved.test_acc
            filename = saveTo .. '/rubiks_best'
            torch.save(filename, saved, 'ascii')
        end

        filename = saveTo .. '/rubiks_epoch' .. epoch
        torch.save(filename, saved, 'ascii')

        -- convert back
        if CUDA then
            model = model:cuda()
        end

        epoch = epoch + 1
    end
end


local from_cmd_line = (debug.getinfo(3).name == nil)

if from_cmd_line then
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text("Training script for Rubik's Cube neural net solve")
    cmd:option('--epslen', 2, 'episode length')
    cmd:option('--savedir', 'models', 'Save directory')
    cmd:option('--type', 'none', 'Model type')
    cmd:option('--ntrain', 50000, 'Number training episodes')
    cmd:option('--ntest', 10000, 'Number testing episodes')
    cmd:option('--epochs', 40, 'Number epochs')
    cmd:option('--batchsize', 8, 'Batch size to use')
    cmd:option('--learningrate', 0.1, 'Learning rate used')
    cmd:option('--gpu', 0, 'Use GPU or not')
    opt = cmd:parse(arg or {})

    CUDA = (opt.gpu ~= 0)
    if CUDA then
        require 'cutorch'
        require 'cunn'
    end

    hyperparams = {
        seed = 987,
        n_epochs = opt.epochs,
        batchSize = opt.batchsize,
        learningRate = opt.learningrate,
        n_train = opt.ntrain,
        n_valid = 1000,
        n_test = opt.ntest,
        hiddenSize = 100,
        rho = 10,
        episode_length = opt.epslen,
        saved_to = opt.savedir,
        model_type = opt.type,
        using_gpu = CUDA
    }
    _setupHyperparams()

    if hyperparams.model_type == 'full' then
        print('Training a fully connected model')
        model, loss = fullyConnected()
    elseif hyperparams.model_type == 'rnn' then
        print('Training a plain recurrent model')
        model, loss = plainRecurrent()
    elseif hyperparams.model_type == 'lstm' then
        print('Training an LSTM')
        model, loss = LSTM()
    elseif hyperparams.model_type == 'fulltwo' then
        print('Training a 2 hidden layer FC model')
        model, loss = biggerFullyConnected()
    else
        print('Invalid model type, exiting')
        return
    end

    if CUDA then
        model = model:cuda()
        loss = loss:cuda()
    end
    print('Finished building model')

    print('Loaded hyperparameters')
    print(hyperparams)
    print('Using episode length', EPISODE_LENGTH)
    print('Saving to', saveTo)

    timer = torch.Timer()

    if model ~= nil then
        trainModel(model, loss)
    end

    seconds = timer:time().real
    minutes = math.floor(seconds / 60)
    seconds = seconds - 60 * minutes
    print(string.format('Took %d minutes %f seconds', minutes, seconds))
end
