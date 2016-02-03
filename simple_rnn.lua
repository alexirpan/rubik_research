require 'rnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('A simple RNN')
cmd:option('--learningRate', 0.1, 'learning rate')
cmd:option('--hiddenSize', 10, 'hidden size')
cmd:option('--batchSize', 8, 'batch size')
cmd:option('--rho', 5, 'sequence length')
cmd:option('--nIndex', 100, 'size of input/output')

opt = cmd:parse(arg or {})

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
rnn = nn.Recursor(rnn, opt.rho)

-- criterion for loss
criterion = nn.ClassNLLCriterion()

-- arbitrary dataset
sequence_ = torch.LongTensor():range(1,10) -- 1 to 10
sequence = torch.LongTensor(100,10) -- initialized
sequence:copy(sequence_:view(1,10):expand(100,10)) -- copies 1 to 10 into the 100 rows
sequence:resize(100*10) -- and flattens

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
    --
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

    local outputs, err = {}, 0
    for step = 1, opt.rho do
        outputs[step] = rnn:forward(inputs[step])
        err = err + criterion:forward(outputs[step], targets[step])
    end

    print(string.format("Iteration %d: Loss = %f", iter, err))

    -- backprop through time
    local gradOutputs, gradInputs = {}, {}
    for step = opt.rho,1,-1 do
        gradOutputs[step] = criterion:backward(outputs[step], targets[step])
        gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])
    end

    -- and finally apply the updates
    rnn:updateParameters(opt.learningRate)
end
