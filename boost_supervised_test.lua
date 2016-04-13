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
    return eps, eps_labels
end


--TODO refactor here too
function testModels(models, n_test, episode_length)
    test, test_labels = generateEpisodes(n_test, episode_length)
    -- test error
    local test_err, test_correct = 0, 0

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
            local total = torch.Tensor(N_MOVES):zero()
            for i = 1, #models do
                total = total + outputs[i][step]:exp()
            end
            _, best = total:max(1)
            if best[1] == target[step] then
                test_correct = test_correct + 1
            end
        end
    end
    -- again account for episode length
    test_correct = test_correct / (n_test * episode_length) * 100
    print(string.format("Test accuracy = %f %%", test_correct))
end


local from_cmd_line = (debug.getinfo(3).name == nil)

if from_cmd_line then
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text("Testing script to measure supervised learning accuracy")
    cmd:text("when using an average of previous models")
    cmd:option('--savedir', 'models', 'Directory where models were saved')
    cmd:option('--nmodels', 10, 'Number of models to average')
    cmd:option('--startepoch', 1, 'Which epoch to start at. Takes all models from startepoch to startepoch + nmodels - 1')
    cmd:option('--ntest', 10000, 'Number episodes to use for accuracy')
    opt = cmd:parse(arg or {})

    torch.manualSeed(12345)
    hyperparams = torch.load(opt.savedir .. '/hyperparams', 'ascii')
    episode_length = hyperparams.episode_length
    basename = opt.savedir .. '/rubiks_epoch'
    n_models = opt.nmodels
    start = opt.startepoch
    models = {}
    for ep = start, start + n_models - 1 do
        table.insert(
            models,
            torch.load(basename .. ep, 'ascii').model
        )
    end
    testModels(models, opt.ntest, episode_length)
end
