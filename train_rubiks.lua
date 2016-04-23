require 'rnn'
require 'rubiks'
require 'rubiks_utils'

require 'adafilter'
require 'averager'


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
    prev_epochs = hyperparams.prev_epochs
    batchSize = hyperparams.batchSize
    learningRate = hyperparams.learningRate
    n_train = hyperparams.n_train
    n_valid = hyperparams.n_valid
    n_test = hyperparams.n_test
    episode_length = hyperparams.episode_length
    saveTo = hyperparams.saved_to
    hiddenSize = hyperparams.hiddenSize
    rho = hyperparams.rho
end


function generateEpisodes(n_episodes)
    local eps = torch.Tensor(n_episodes * episode_length, N_STICKERS, N_COLORS):zero()
    -- using a LongTensor makes later comparison easier. These are class indices so it's fine
    local eps_labels = torch.LongTensor(n_episodes * episode_length):zero()

    for i = 1, n_episodes do
        local episode, moves = randomCubeEpisode(episode_length)
        local start = (i-1) * episode_length
        eps[{ {start+1, start+episode_length} }] = episode
        eps_labels[{ {start+1, start+episode_length} }] = moves
    end

    if CUDA then
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

    return fcnn
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

    return fcnn
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

    return rnn
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

    return lstm
end


function lossFn()
    -- Every model type uses the same loss function
    -- Decoupling the loss constructor from the model constructor
    -- lets us create loss functions for previously saved models.
    --
    -- (It also lets us more easily try different loss functions in
    -- the future.)
    local loss = nn.SequencerCriterion(nn.ClassNLLCriterion())
    return loss
end


-- The intention with these two functions is to keep all the CSV
-- writing information in the same place
function csvHeader()
    return 'epoch,train_err,train_acc,test_err,test_acc,total_time\n'
end


function csvLine(save_info)
    return string.format('%d,%f,%f,%f,%f,%f\n',
        save_info.epoch,
        save_info.train_err,
        save_info.train_acc,
        save_info.test_err,
        save_info.test_acc,
        save_info.total_time
    )
end


function csvAdaHeader()
    return 'epoch,train_err,train_acc,pseudoloss,unweighted_good_ploss,unweighted_bad_ploss,boost_acc,boost_time,total_time\n'
end


function csvAdaLine(save_info)
    return string.format('%d,%f,%f,%f,%f,%f,%f,%f,%f\n',
        save_info.epoch,
        save_info.train_err,
        save_info.train_acc,
        save_info.pseudoloss,
        save_info.avg_unweighted_good_ploss,
        save_info.avg_unweighted_bad_ploss,
        save_info.boost_acc,
        save_info.boost_time,
        save_info.total_time
    )
end


function lastTimeFromCsv(file)
    --Reads total time from the final csv row
    --When function finishes, file pointer will be at end of file
    file:seekEnd()
    local endpos = file:position()
    file:seek(1)

    --Read lines until we hit the end
    local line = nil
    while file:position() < endpos do
        line = file:readString('*l')
    end

    --split line on commas
    local tokens = {}
    for tok in string.gmatch(line, '[^,]+') do
        table.insert(tokens, tok)
    end
    return tonumber(tokens[6])
end


function trainModel(model, loss)
    local timer = torch.Timer()
    local timeOffset = 0

    best_acc = 0
    -- It's structured this way to make sure that if the script is run for
    -- 20 epochs, and the same model is trained for another 20 epochs, all
    -- saved information for the second run starts at 21 (and ends at 40)
    epoch = prev_epochs + 1

    -- this file stays open until end of training
    -- Always open this file in rw mode! Sometimes we run the training script on
    -- the same directory (ex: when we want to refine an already trained model)
    -- and in that case the training csv will already exist. Opening in w mode
    -- auto deletes all data
    trainCsv = torch.DiskFile(hyperparams.saved_to .. '/trainingdata.csv', 'rw')
    trainCsv:seekEnd()
    if trainCsv:position() == 1 then
        -- writes header only if file is initially empty
        trainCsv:writeString(csvHeader())
    else
        -- find time offset
        timeOffset = lastTimeFromCsv(trainCsv)
    end

    while epoch <= n_epochs do
        print('Starting epoch', epoch)
        print('Creating data')
        local dataTimer = torch.Timer()

        data = createDataset(n_train, n_valid, n_test)
        -- flatten last two axes
        data['train']:resize(n_train * episode_length,
                             N_STICKERS * N_COLORS)
        data['valid']:resize(n_test * episode_length,
                             N_STICKERS * N_COLORS)
        data['test']:resize(n_test * episode_length,
                             N_STICKERS * N_COLORS)

        seconds = dataTimer:time().real
        minutes = math.floor(seconds / 60)
        seconds = seconds - 60 * minutes
        print(string.format('Spent %d minutes %f seconds creating data', minutes, seconds))

        train = data['train']
        train_labels = data['train_labels']
        valid = data['valid']
        valid_labels = data['valid_labels']
        test = data['test']
        test_labels = data['test_labels']

        local err, correct = 0, 0

        -- go through batches in random order
        local nBatches = n_train / batchSize
        local perm = torch.randperm(nBatches)

        -- n_train is the number of sequences
        -- we do (batchSize) sequences at once
        for j = 1, nBatches do
            ind = perm[j]
            -- generate a batch of sequences
            -- as constructed, each episode_length block is a new sequence
            -- So to get batchSize sequences at once, we need to start pulling from
            -- (start, start + EPISODE, start + 2*EPISODE, ...)
            start = (ind - 1) * batchSize * episode_length + 1
            local seqIndices = torch.LongTensor():range(
                start, start + (batchSize - 1) * episode_length, episode_length
            )

            local inputs, targets = {}, {}
            for step = 1, episode_length do
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
                start, start + (batchSize - 1) * episode_length, episode_length
            )

            for step = 1, episode_length do
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
        -- for averages, we need to account for there being n_episodes * episode_length samples total
        -- Loss already averages over batch size, so we only need to divide by
        -- number of batches
        err = err / (nBatches * episode_length)
        -- On the other hand, here we do need to divide by the number of training
        -- examples
        correct = correct / (n_train * episode_length) * 100
        print(string.format(
            "Epoch %d: Average training loss = %f, training accuracy = %f %%",
            epoch,
            err,
            correct
        ))

        -- test error
        -- Since we're not interested in updating the model, we can run one
        -- huge batch
        local test_err, test_correct = 0, 0
        -- Construct table from testset
        local seqIndices = torch.LongTensor():range(
            1, 1 + (n_test-1) * episode_length, episode_length
        )
        local inputs = {}
        for step = 1, episode_length do
            inputs[step] = test:index(1, seqIndices)
            seqIndices = seqIndices + 1
        end
        model:forget() -- forget past test runs
        local outputs = model:forward(inputs)
        local seqIndices = torch.LongTensor():range(
            1, 1 + (n_test-1) * episode_length, episode_length
        )
        local targets = {}
        for step = 1, episode_length do
            targets[step] = test_labels:index(1, seqIndices)
            seqIndices = seqIndices + 1
        end
        test_err = loss:forward(outputs, targets)

        -- outputs is a table of seqlen entries, each of size
        -- (batchsize, N_MOVES), and here batchsize = n_test
        for step = 1, episode_length do
            _, best = outputs[step]:max(2)
            best:resize(n_test)
            trueVal = targets[step]
            test_correct = test_correct + best:eq(trueVal):sum()
        end
        -- again account for episode length
        -- TODO verify that the loss is automatically averaged over the batch size
        -- (This appears to be the case but I can't find it in the docs)
        test_err = test_err / episode_length
        test_correct = test_correct / (n_test * episode_length) * 100
        print(string.format("Test loss = %f, test accuracy = %f %%", test_err, test_correct))

        -- save epoch data
        if CUDA then
            -- convert the model back to DoubleTensor before saving
            -- Otherwise it's only loadable on machines with GPU
            model = model:double()
        end

        total_time_sec = timer:time().real
        total_time_min = total_time_sec / 60

        -- TODO
        -- If models take up too much space, only save the model with highest
        -- validation accuracy
        saved = {
            model = model,
            train_err = err,
            train_acc = correct,
            test_err = test_err,
            test_acc = test_correct,
            epoch = epoch,
            total_time = total_time_min + timeOffset,
            hyperparams = hyperparams
        }
        trainCsv:writeString(csvLine(saved))

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


-- It's happening.
-- The dreaded method that is completely modified in another branch
-- One day, this will be less awful. But not today
-- Key differences should be highlighted by the comments
function trainModelAdaBoost(weak_model, loss)
    assert(hyperparams.batchSize == 1, "AdaBoost requires batch size 1")
    -- In this method, "model" is the weak learner
    -- It will be copied to be used in the final booster
    local timer = torch.Timer()
    local timeOffset = 0

    -- DIFF ONE
    -- More hyperparameters, define a new model
    hyperparams.max_models = 10
    -- Resave the hyperparams
    torch.save(opt.savedir .. '/hyperparams', hyperparams, 'ascii')
    local model = nn.Averager({}, 0, hyperparams.max_models)
    if CUDA then
        model = model:cuda()
    end
    -- Note we don't need to wrap Averager in Sequencer, because this
    -- assumes all models given to it are Sequencers, so they will
    -- unpack tables automatically
    -- END DIFF ONE

    best_acc = 0
    -- It's structured this way to make sure that if the script is run for
    -- 20 epochs, and the same model is trained for another 20 epochs, all
    -- saved information for the second run starts at 21 (and ends at 40)
    epoch = prev_epochs + 1

    -- this file stays open until end of training
    -- Always open this file in rw mode! Sometimes we run the training script on
    -- the same directory (ex: when we want to refine an already trained model)
    -- and in that case the training csv will already exist. Opening in w mode
    -- auto deletes all data
    trainCsv = torch.DiskFile(hyperparams.saved_to .. '/trainingdata.csv', 'rw')
    trainCsv:seekEnd()
    -- DIFF 1.5
    -- Special CSV writing for AdaBoost
    if trainCsv:position() == 1 then
        -- writes header only if file is initially empty
        trainCsv:writeString(csvAdaHeader())
    else
        -- find time offset
        timeOffset = lastTimeFromCsv(trainCsv)
    end
    -- END DIFF 1.5

    -- DIFF 1.75
    -- It makes most sense to place dataset regeneration at the end of each epoch
    -- So, sample the data initially here
    print('Initializing dataset')
    local train, train_labels = generateEpisodes(n_train)
    train:resize(n_train * episode_length,
                 N_STICKERS * N_COLORS)
    local train_weights = torch.ones(n_train)
    if CUDA then
        train_weights = train_weights:cuda()
    end

    while epoch <= n_epochs do
        print('Starting epoch', epoch)
        -- Sometimes we get some accumulated floating point errors, this helps
        -- fix some of that
        train_weights = train_weights * n_train / train_weights:sum()
        if math.abs(train_weights:sum() - n_train) < 0.001 then
            print(string.format('!! Expected weights to sum to %d !!', n_train))
            print('Actually sum to', train_weights:sum())
        end
        local err, correct = 0, 0
        -- Will reorder later, this makes it easier to collect
        local allTrainOutputs = torch.Tensor(n_train * episode_length, N_MOVES):zero()
        if CUDA then
            allTrainOutputs = allTrainOutputs:cuda()
        end
        -- DIFF THREE
        -- Changing all of this code to use weak model instead, and saving all
        -- model outputs for later computation
        -- go through batches in random order
        local nBatches = n_train / batchSize
        local perm = torch.randperm(nBatches)

        -- ANOTHER BIG DIFF HERE
        -- We want to weight each episode differently.
        -- However, we're running in batch, and each episode in a batch may have
        -- different weight.
        -- The default ClassNLLCriterion lets you reweight classes for computing
        -- loss, but not reweight specific examples.
        --
        -- Doing this properly will require hacking something in both the
        -- SequencerCriterion and ClassNLLCriterion, which is more engineering
        -- than I have time for.
        --
        -- Instead, we enforce batchSize 1. When this is true, the scaling is
        -- constant over the batch
        --
        -- n_train is the number of sequences
        -- we do (batchSize) sequences at once
        for j = 1, nBatches do
            ind = perm[j]
            -- generate a batch of sequences
            -- as constructed, each episode_length block is a new sequence
            -- So to get batchSize sequences at once, we need to start pulling from
            -- (start, start + EPISODE, start + 2*EPISODE, ...)
            start = (ind - 1) * batchSize * episode_length + 1
            local seqIndices = torch.LongTensor():range(
                start, start + (batchSize - 1) * episode_length, episode_length
            )

            local inputs, targets = {}, {}
            for step = 1, episode_length do
                inputs[step] = train:index(1, seqIndices)
                targets[step] = train_labels:index(1, seqIndices)
                seqIndices:add(1)
            end

            -- run sequence forward through rnn
            weak_model:zeroGradParameters()
            weak_model:forget()  -- forget past time steps

            local outputs = weak_model:forward(inputs)
            -- Here is where we use assumption that ind is index into episode number
            err = err + train_weights[ind] * loss:forward(outputs, targets)

            -- reset seqIndices to check accuracy
            seqIndices = torch.LongTensor():range(
                start, start + (batchSize - 1) * episode_length, episode_length
            )

            for step = 1, episode_length do
                -- compute accuracy
                -- shape of output at each step is (batchSize, N_MOVES)
                _, predicted = outputs[step]:max(2)
                predicted:resize(batchSize)
                trueVal = targets[step]
                -- batchSize 1 assumption also used here
                -- (Need weighted accuracy for both error and classifcation)
                if predicted[1] == trueVal[1] then
                    correct = correct + train_weights[ind]
                end
                -- Copy outputs into allTrainOutputs
                local batchStart = (ind - 1) * batchSize * episode_length
                -- TODO fix this when batchSize > 1
                allTrainOutputs[batchStart + step] = outputs[step]
            end

            -- backward sequence (backprop through time)
            local gradOutputs = loss:backward(outputs, targets)
            local gradInputs = weak_model:backward(inputs, gradOutputs)

            -- and update
            -- batchSize 1 assumption also used here
            weak_model:updateParameters(train_weights[ind] * learningRate)
        end
        -- END DIFF THREE
        -- for averages, we need to account for there being n_episodes * episode_length samples total
        -- Loss already averages over batch size, so we only need to divide by
        -- number of batches
        err = err / (nBatches * episode_length)
        -- On the other hand, here we do need to divide by the number of training
        -- examples
        correct = correct / (n_train * episode_length) * 100
        print(string.format(
            "Epoch %d: Average training loss = %f, training accuracy = %f %%",
            epoch,
            err,
            correct
        ))

        -- DIFF SIX
        -- Compute edge of the weak_model and add to boosted model
        -- (outputs are from weak model, see above)
        allTrainOutputs = allTrainOutputs:exp()  -- Expects probabilities, not log probs

        local plosses, unweighted_good, unweighted_bad = pseudoloss(allTrainOutputs, train_labels, n_train, episode_length)
        local weighted_losses = plosses * train_weights
        -- !! Torch multiplication of two 1D tensors does dot product !!
        local avg_weighted_loss = plosses * train_weights / n_train

        print(string.format("Pseudoloss over D_%d = %f", epoch, avg_weighted_loss))

        if avg_weighted_loss > 0.5 then
            print("!!! WORSE THAN RANDOM GUESSING !!!")
            print("Not adding classifier this round, trying again...")
            local update_weights = false
        else
            local alpha = math.log( (1 - avg_weighted_loss) / avg_weighted_loss )
            local toAdd = weak_model:clone()
            model:addModel(weak_model, alpha)
            local update_weights = true
        end
        -- END DIFF SIX

        -- DIFF SEVEN
        -- Update datasets for the next epoch
        local next_train = torch.Tensor(n_train * episode_length, N_STICKERS * N_COLORS):zero()
        local next_train_labels = torch.LongTensor(n_train * episode_length):zero()
        local next_train_weights = torch.Tensor(n_train):zero()
        if CUDA then
            next_train = next_train:cuda()
            next_train_labels = next_train_labels:cuda()
            next_train_weights = next_train_weights:cuda()
        end

        if update_weights then
            train_weights = nextWeights(train_weights, plosses, alpha, n_train)
        end

        -- Keep each sample with uniform probability
        local j = 1
        for i = 1, n_train do
            if torch.uniform() < hyperparams.keepprob then
                -- Add the sample
                local start_old = (i-1) * episode_length
                local start_next = (j-1) * episode_length
                next_train[{ {start_next+1, start_next+episode_length} }] = train[{ {start_old+1, start_old+episode_length} }]
                next_train_labels[{ {start_next+1, start_next+episode_length} }] = train_labels[{ {start_old+1, start_old+episode_length} }]
                next_train_weights[j] = train_weights[i]
                j = j + 1
            end
        end
        -- END DIFF SEVEN

        -- DIFF EIGHT
        -- Fill in the gaps, using them to evaluate the power of the boosted model
        local boostTimer = torch.Timer()
        local n_boost_eps = n_train - (j - 1)
        local weight_to_replace = n_train - next_train_weights:sum()

        local boost_eps, boost_labels = generateEpisodes(n_boost_eps)
        boost_eps:resize(n_boost_eps * episode_length, N_STICKERS * N_COLORS)

        -- Always happens even if no weight update (will just use most recent
        -- boosted classifier)
        local boost_weights, boost_correct = computeBoostedWeightsAndAcc(
            model,
            boost_eps,
            boost_labels,
            n_boost_eps,
            episode_length,
            weight_to_replace
        )

        local start = (j-1) * episode_length
        next_train[{ {start+1, n_train * episode_length} }] = boost_eps
        next_train_labels[{ {start+1, n_train * episode_length} }] = boost_labels
        next_train_weights[{ {j, n_train} }] = boost_weights

        boost_correct = boost_correct / (n_boost_eps * episode_length) * 100
        print(string.format("Boosted test accuracy = %f %%", boost_correct))

        train = next_train
        train_labels = next_train_labels
        train_weights = next_train_weights

        seconds = boostTimer:time().real
        boost_time_min = seconds / 60
        seconds = seconds - 60 * math.floor(boost_time_min)
        print(string.format('%d minutes %f seconds refilling data and computing boost accuracy', math.floor(boost_time_min), seconds))
        -- END DIFF SEVEN

        -- save epoch data
        if CUDA then
            -- convert the model back to DoubleTensor before saving
            -- Otherwise it's only loadable on machines with GPU
            model = model:double()
        end

        total_time_sec = timer:time().real
        total_time_min = total_time_sec / 60

        -- DIFF EIGHT
        -- Add Boost info to saved CSV
        -- TODO
        -- If models take up too much space, only save the model with highest
        -- validation accuracy
        saved = {
            model = model,
            train_err = err,
            train_acc = correct,
            pseudoloss = avg_weighted_loss,
            avg_unweighted_good_ploss = unweighted_good,
            avg_unweighted_bad_ploss = unweighted_bad,
            boost_acc = boost_correct,
            epoch = epoch,
            total_time = total_time_min + timeOffset,
            boost_time = boost_time_min,
            hyperparams = hyperparams
        }
        trainCsv:writeString(csvAdaLine(saved))

        if saved.boost_acc > best_acc then
            best_acc = saved.boost_acc
            filename = saveTo .. '/rubiks_best'
            torch.save(filename, saved, 'ascii')
        end
        -- END DIFF EIGHT

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
    local NOMODEL = 'NOMODEL'
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
    cmd:option('--model', NOMODEL, "Initialize with a pre-trained model. Note this will clobber the model type (but will not clobber anything else)")
    cmd:option('--adaboost', 0, 'Use AdaBoost or not')
    cmd:option('--keepprob', 0.9, 'Probability of keeping sample for next iteration')
    opt = cmd:parse(arg or {})

    CUDA = (opt.gpu ~= 0)
    if CUDA then
        require 'cutorch'
        require 'cunn'
    end

    hyperparams = {
        seed = 987,
        prev_epochs = 0,
        n_epochs = opt.epochs,
        batchSize = opt.batchsize,
        learningRate = opt.learningrate,
        n_train = opt.ntrain,
        n_valid = 0, -- the way this is set up, the test set is actually the validation set
        n_test = opt.ntest,
        hiddenSize = 100,
        rho = 10, -- I believe this is overridden by the rnn library, so ignore this
        episode_length = opt.epslen,
        saved_to = opt.savedir,
        model_type = opt.type,
        using_gpu = CUDA,
        adaboost = (opt.adaboost ~= 0),
        keepprob = opt.keepprob
    }
    if hyperparams.adaboost then
        assert(hyperparams.batchSize == 1, 'AdaBoost only supports batch size 1 for now')
    end

    if opt.model ~= NOMODEL then
        print('Loading a previously trained model')
        print('Loading from ' .. opt.model .. ' ...')
        data = torch.load(opt.model, 'ascii')
        hyperparams.initial_model = opt.model
        hyperparams.model_type = data.hyperparams.model_type
        hyperparams.prev_epochs = data.epoch
        hyperparams.n_epochs = hyperparams.n_epochs + data.epoch
        --If we keep the seed the same, we'll see the same data as before,
        --so modify it
        hyperparams.seed = hyperparams.seed + 1
    end
    _setupHyperparams()

    if opt.model ~= NOMODEL then
        model = data.model
    elseif hyperparams.model_type == 'full' then
        print('Training a fully connected model')
        model = fullyConnected()
    elseif hyperparams.model_type == 'rnn' then
        print('Training a plain recurrent model')
        model = plainRecurrent()
    elseif hyperparams.model_type == 'lstm' then
        print('Training an LSTM')
        model = LSTM()
    elseif hyperparams.model_type == 'fulltwo' then
        print('Training a 2 hidden layer FC model')
        model = biggerFullyConnected()
    else
        print('Invalid model type, exiting')
        return
    end
    loss = lossFn()

    if CUDA then
        model = model:cuda()
        loss = loss:cuda()
    end
    -- Saving hyperparams
    torch.save(opt.savedir .. '/hyperparams', hyperparams, 'ascii')

    print('Finished building model')

    print('Loaded hyperparameters')
    print(hyperparams)
    print('Using episode length', episode_length)
    print('Saving to', saveTo)

    timer = torch.Timer()

    if model ~= nil then
        if hyperparams.adaboost then
            trainModelAdaBoost(model, loss)
        else
            trainModel(model, loss)
        end
    end

    seconds = timer:time().real
    minutes = math.floor(seconds / 60)
    seconds = seconds - 60 * minutes
    print(string.format('Took %d minutes %f seconds', minutes, seconds))
end
