-------------------------------------------------------
--[[ Averager ]]--
-- This expects a table of models, all of which output
-- log probabilities of confidence for each class, all of
-- which have the same output shape of 1 x K (where K is
-- number of classes)
--
-- This outputs the log probabilities of the uniform mixture
-- of those distributions. Formally, if there are T distributions,
-- the output is a 1 X K vector where the i-th entry is
--      log( 1/T * \sum_{j=1}^T Pr(model j predicts class i) )
--
-- For the intended use case (averaging models at different epochs for
-- boosting), the notion of gradient is a bit loose since some
-- model parameters are meant to be frozen in time...may be
-- implemented later
--
-- For efficiency reasons, we can't have the booster use
-- a sum of classifiers over all iterations, so we instead
-- use a simple pruning that keeps only the classifiers
-- with largest weights so far
-------------------------------------------------------
require 'rnn'
require 'rubiks'
local Averager, parent = torch.class("nn.Averager", "nn.Module")


function Averager:__init(models, weights, max_models)
    -- Weights is assumed to be a Tensor, not a table!
    parent.__init(self)
    if #models == 0 then
        -- Do initialization later
        self.models = {}
        self.n_models = 0
        self.max_models = max_models
        return
    end
    assert(torch.isTensor(weights))
    assert(#models == weights:size(1))
    assert(#models <= max_models)
    self.models = models
    self.n_models = #models
    self.max_models = max_models
    self.weights = weights
end


function Averager:_sortModels()
    -- Sorts models by weights, with smallest weight last
    assert(self.n_models > 0)
    local sorted, indices = self.weights:sort(1, true)
    self.weights = sorted
    local temp = {}
    for i = 1, self.n_models do
        table.insert(temp, self.models[indices[i]])
    end
    self.models = temp
end


function Averager:addModel(model, weight)
    if self.n_models == 0 then
        self.weights = torch.Tensor(1)
        self.weights[1] = weight
    else
        local new_weights = torch.Tensor(self.n_models + 1)
        new_weights[{ {1, self.n_models} }] = self.weights
        new_weights[self.n_models + 1] = weight
        self.weights = new_weights
    end

    table.insert(self.models, model)
    self.n_models = self.n_models + 1

    self:_sortModels()

    if self.n_models > self.max_models then
        table.remove(self.models)
        self.weights = self.weights[{ {1, self.max_models} }]
        self.n_models = self.n_models - 1
    end
end


function Averager:updateOutput(input)
    -- given a table of output probabilities
    if self.n_models == 0 then
        -- TODO don't use global number of classes here
        local out = torch.ones(N_MOVES) / N_MOVES
        return out
    end
    self.model_outputs = {}
    for i=1, self.n_models do
        self.model_outputs[i] = self.models[i]:forward(input)
    end

    local n_classes = #self.model_outputs[1]
    local total = torch.Tensor(n_classes):zero()
    for i=1, self.n_models do
        -- output is log prob, copy to keep model output intact
        total = total + self.model_outputs[i]:clone():exp() * self.weights[i]
    end
    total = total / self.weights:sum()
    return total
end


-- Helper methods to make it easy to reset state over all modesl
function Averager:zeroGradParameters()
    for i = 1, self.n_models do
        self.models[i]:zeroGradParameters()
    end
end


function Averager:forget()
    for i = 1, self.n_models do
        self.models[i]:forget()
    end
end


function Averager:double()
    for i = 1, self.n_models do
        self.models[i] = self.models[i]:double()
    end
end


function Averager:cuda()
    for i = 1, self.n_models do
        self.models[i] = self.models[i]:cuda()
    end
end


function Averager:updateGradInput(probTable, gradOutput)
    --TODO decide if this is worth implementing
    error("Averager gradient update not implemented")
end


