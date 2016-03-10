CUDA = false

if CUDA then
    require 'cutorch'
    require 'cunn'
end

require 'rnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('A simple RNN')
cmd:option('--learningRate', 0.1, 'learning rate')
cmd:option('--hiddenSize', 300, 'hidden size')
cmd:option('--batchSize', 8, 'batch size')
cmd:option('--rho', 5, 'sequence length')
cmd:option('--nIndex', 100, 'size of input/output')

opt = cmd:parse(arg or {})

if CUDA then
    print('Using GPU')
else
    print('Using CPU')
end

timer = torch.Timer()

recur = nn.Recurrent(
    opt.hiddenSize,                             -- output size
    nn.LookupTable(opt.nIndex, opt.hiddenSize), -- input
    nn.Linear(opt.hiddenSize, opt.hiddenSize),  -- feedbacks previous output tensor to transfer
    nn.Sigmoid(),                               -- transfer, nonlinearity applied to sum of input and feedback
    opt.rho                                     -- backprop steps to go back in time
)

local rnn = nn.Sequential()
rnn:add(recur)
rnn:add(nn.Linear(opt.hiddenSize, opt.nIndex))
rnn:add(nn.LogSoftMax())

-- wrap the non-recurrent Sequential module into Recursor
-- wrap rnn with a Sequencer (lets us pass sequences directly as input/output)
rnn = nn.Sequencer(rnn)
if CUDA then rnn:cuda() end

-- criterion for loss
-- (needs to be modified)
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
if CUDA then criterion:cuda() end

-- arbitrary dataset
sequence_ = torch.LongTensor():range(1,10) -- 1 to 10
sequence = torch.LongTensor(100,10) -- initialized
sequence:copy(sequence_:view(1,10):expand(100,10)) -- copies 1 to 10 into the 100 rows
sequence:resize(100*10) -- and flattens

if CUDA then sequence = sequence:cuda() end

offsets = {}
for i = 1, opt.batchSize do
    table.insert(offsets, math.ceil(math.random() * opt.batchSize))
end
offsets = torch.LongTensor(offsets)

-- training
iter = 1
max_iters = math.pow(10, 3)
while iter < max_iters do
    iter = iter + 1
    -- sequence of rho time steps
    local inputs, targets = {}, {}
    for step = 1, opt.rho do
        -- get input batch
        inputs[step] = sequence:index(1, offsets)
        -- increment indices
        -- (the % operator is very slow in Lua, so avoid it in tight loops)
        offsets:add(1)
        for j = 1, opt.batchSize do
            if offsets[j] > opt.nIndex then
                offsets[j] = 1
            end
        end
        targets[step] = sequence:index(1, offsets)
    end

    -- forward sequence
    rnn:zeroGradParameters()
    rnn:forget() -- forget prev time steps

    local outputs = rnn:forward(inputs)
    local err = criterion:forward(outputs, targets)

    print(string.format("Iteration %d: Loss = %f", iter, err))

    -- backprop through time
    local gradOutputs = criterion:backward(outputs, targets)
    local gradInputs = rnn:backward(inputs, gradOutputs)

    -- and apply update
    rnn:updateParameters(opt.learningRate)
end

print('Time elapsed: ' .. timer:time().real .. ' seconds')
