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
-------------------------------------------------------
local Averager, parent = torch.class("nn.Averager", "nn.Module")


function Averager:__init(models)
    parent.__init(self)
    self.models = models
    self.n_models = #models
end


function Averager:updateOutput(input)
    -- given a table of output probabilities
    self.model_outputs = {}
    for i=1, self.n_models do
        self.model_outputs[i] = self.models[i]:forward(input)
    end

    local n_classes = #self.model_outputs[1]
    local total = torch.Tensor(n_classes):zero()
    for i=1, self.n_models do
        -- output is log prob, copy to keep model output intact
        total = total + self.model_outputs[i]:clone():exp()
    end
    total = total / self.n_models
    return total:log()
end


function Averager:updateGradInput(probTable, gradOutput)
    --TODO decide if this is worth implementing
    error("Averager gradient update not implemented")
end


