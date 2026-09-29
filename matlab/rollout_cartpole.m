function [obs, ret] = rollout_cartpole(agent, env, actionNoiseStd, maxSteps)
%ROLLOUT_CARTPOLE Step CartPole with an agent's deterministic policy.
%   [obs, ret] = rollout_cartpole(agent, env, actionNoiseStd, maxSteps)
%   returns the visited observations (T x obsDim) and the episode return.
%   Gaussian noise is added to the action to drive the pole away from the
%   equilibrium and visit a broader set of states.
%
%   This steps the environment directly instead of calling sim(), because sim()
%   requires the agent's observation spec to equal the environment's, and the
%   deployed agent carries observation limits that the environment does not.

    if nargin < 3, actionNoiseStd = 0;  end
    if nargin < 4, maxSteps = 500;      end

    limit = getActionInfo(env).UpperLimit;
    x     = reset(env);
    obs   = zeros(0, numel(x));
    ret   = 0;
    for t = 1:maxSteps
        obs(end+1, :) = x(:).'; %#ok<AGROW>
        a = getAction(agent, {x});
        a = a{1} + actionNoiseStd * randn(size(a{1}));
        a = max(min(a, limit), -limit);
        [x, r, isDone] = step(env, a);
        ret = ret + r;
        if isDone, break; end
    end
end
