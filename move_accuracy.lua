--Outputs data that says how accurate a model is at predicting a move
--K moves out. Using this to test my hypothesis that accuracy drops
--with large episode lengths because the dataset assumes one single correct
--move when there could be several

require 'rnn'
require 'rubiks' -- drop in appropriate constants
require 'rubiks_utils'


function generatePredictionData(model, num_episodes, length)
    feats = torch.Tensor(num_episodes * length, N_STICKERS, N_COLORS):zero()
    labels = torch.LongTensor(num_episodes * length):zero()
    for i = 1, num_episodes do
        episode, moves = randomCubeEpisode(length)
        start = (i-1) * length
        feats[{ {start+1, start+length} }] = episode
        labels[{ {start+1, start+length} }] = moves
    end
    feats:resize(num_episodes * length, N_STICKERS * N_COLORS)
    print('Generated dataset')

    correct = torch.Tensor(length):zero()

    for i = 1, num_episodes do
        local start = (i-1) * length + 1
        input_ = feats:narrow(1, start, length)
        -- from Tensor to table
        input = {}
        for step = 1, length do
            input[step] = input_[step]
        end
        output = model:forward(input)
        target = labels:narrow(1, start, length)
        for step = 1, length do
            _, best = output[step]:max(1)
            -- unwrap best from table
            if target[step] == best[1] then
                -- Note episodes are in backwards order!
                correct[length - step + 1] = correct[length - step + 1] + 1
            end
        end

        if i % 100 == 0 then
            print(i .. ' episodes done')
        end
    end

    correct = correct / num_episodes
    return correct
end


-- It is expected that this is imported by itself for use elsewhere.
-- e.g. in an iTorch notebook
