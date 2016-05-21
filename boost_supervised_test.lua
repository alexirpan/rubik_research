--[[
    Computes supervised test accuracy of an average of models
--]]
require 'rnn'
require 'train_rubiks'
require 'rubiks_utils'


--TODO refactor this for less copy-pasting
function generateEpisodes(n_episodes, episode_length)
    local eps = torch.Tensor(n_episodes * episode_length, N_STICKERS, N_COLORS):zero()
    -- using a LongTensor makes later comparison easier. These are class indices so it's fine
    local eps_labels = torch.LongTensor(n_episodes * episode_length):zero()

    for i = 1, n_episodes do
        local episode, moves = randomCubeEpisode(episode_length)
        local start = (i-1) * episode_length
        eps[{ {start+1, start+episode_length} }] = episode
        eps_labels[{ {start+1, start+episode_length} }] = moves
    end
    eps:resize(n_episodes * episode_length,
               N_STICKERS * N_COLORS)

    if CUDA then
        eps = eps:cuda()
        eps_labels = eps_labels:cuda()
    end

    return eps, eps_labels
end


-- Old version of the code expected this to average the given models
-- Cannibalizing it to print accuracy of each individual model
function testModels(models, n_test, episode_length)
    test, test_labels = generateEpisodes(n_test, episode_length)
    -- test error
    local test_correct = torch.zeros(#models)

    for i = 1, n_test do
        if i % 100 == 0 then
            print(n_test - i, 'episodes left')
        end
        local start = (i-1) * episode_length + 1
        input_ = test:narrow(1, start, episode_length)
        -- The sequencer interface expects a Lua table of
        -- seqlen entries, each of which is a batchsize x featsize tensor
        input = {}
        for step = 1,episode_length do
            input[step] = input_[step]
        end
        outputs = {}
        for i = 1, #models do
            models[i]:forget()
            outputs[i] = models[i]:forward(input)
        end

        target = test_labels:narrow(1, start, episode_length)
        for step = 1, episode_length do
            for i = 1, #models do
                _, best = outputs[i][step]:max(1)
                if best[1] == target[step] then
                    test_correct[i] = test_correct[i] + 1
                end
            end
        end
    end
    -- again account for episode length
    test_correct = test_correct / (n_test * episode_length) * 100
    return test_correct
end


-- Also cannibalizing this code
local from_cmd_line = (debug.getinfo(3).name == nil)

if from_cmd_line then
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text("Testing script to measure supervised learning accuracy")
    cmd:text("when using an average of previous models")
    NOMODEL = 'NOMODEL'
    cmd:option('--model', NOMODEL, 'Boosted model to open')
    cmd:option('--savedir', 'models', 'Directory where models were saved')
    cmd:option('--nmodels', 10, 'Number of models to average')
    cmd:option('--startepoch', 1, 'Which epoch to start at. Takes all models from startepoch to startepoch + nmodels - 1')
    cmd:option('--ntest', 10000, 'Number episodes to use for accuracy')
    cmd:option('--gpu', 0, 'Use GPU or not')
    opt = cmd:parse(arg or {})

    if opt.model == NOMODEL then
        print("No model specified")
        return
    end

    CUDA = (opt.gpu ~= 0)
    if CUDA then
        require 'cutorch'
        require 'cunn'
    end

    torch.manualSeed(12345)
    if CUDA then
        cutorch.manualSeedAll(12345)
    end

    hyperparams = torch.load(opt.savedir .. '/hyperparams', 'ascii')
    episode_length = hyperparams.episode_length

    name = opt.savedir .. '/' .. opt.model

    print("Loading model...")
    data = torch.load(name, 'ascii')
    model = data.model
    if CUDA then
        model = model:cuda()
    end
    print("Model loaded")

    acc = testModels(model.models, opt.ntest, episode_length)
    print("Model accuracies")
    print(acc)
    best, ind = acc:max(1)
    best = best[1]
    ind = ind[1]
    print(string.format("Best model is model %d with acc %f", ind, best))
    best_model = model.models[ind]
    if CUDA then
        best_model = best_model:double()
    end
    torch.save(opt.savedir .. '/best_individual_model', best_model, 'ascii')
end
