%% s2_generate_validation_data.m
% Creates the reference data the Python side is validated against: observations
% and the actions MATLAB's getAction returns for them.
%
% Two data sets are written to artifacts/validation/:
%   box_*         uniform samples over a box around the operating range
%                 (covers the whole input space, including extreme values)
%   trajectory_*  observations visited while simulating the agent with
%                 exploration noise (realistic input distribution)
%
% Each set is a pair <name>_obs.csv (N x obsDim) / <name>_action.csv (N x actDim).

clear; clc;

root   = fileparts(fileparts(mfilename('fullpath')));
outDir = fullfile(root, "artifacts", "validation");
if ~isfolder(outDir), mkdir(outDir); end

% ---- configuration ---------------------------------------------------------
% CartPole observations are unbounded (+-Inf), so the sampling box is chosen
% by hand: [x; dx; theta; dtheta]. theta is +-0.21 rad (the failure angle).
sampleLow  = [-2.4; -3; -0.21; -3];
sampleHigh = [ 2.4;  3;  0.21;  3];
nBox        = 300;
nTrajEpisodes = 10;
% ----------------------------------------------------------------------------

S = load(fullfile(root, "artifacts", "agent.mat"));
agent = S.agent;
agent.UseExplorationPolicy = false;   % deterministic, as deployed

obsDim = getObservationInfo(agent).Dimension(1);
actDim = getActionInfo(agent).Dimension(1);
assert(numel(sampleLow) == obsDim && numel(sampleHigh) == obsDim, ...
    "sampleLow/sampleHigh must have obsDim = %d entries.", obsDim);

% ---- box samples -----------------------------------------------------------
rng(0);
Obox = sampleLow.' + rand(nBox, obsDim) .* (sampleHigh - sampleLow).';
writeSet(agent, Obox, fullfile(outDir, "box"));

% ---- trajectory samples ----------------------------------------------------
% Simulate with exploration ON so the trajectories leave the equilibrium and
% visit a broader set of states. The reference actions below are still
% computed with the deterministic policy.
env = rlPredefinedEnv("CartPole-Continuous");
agent.UseExplorationPolicy = true;
Otraj = zeros(0, obsDim);
for k = 1:nTrajEpisodes
    out = sim(env, agent, rlSimulationOptions(MaxSteps = 500));
    ch  = fieldnames(out.Observation);
    d   = squeeze(out.Observation.(ch{1}).Data);      % obsDim x T
    Otraj = [Otraj; d.']; %#ok<AGROW>
end
agent.UseExplorationPolicy = false;
writeSet(agent, Otraj, fullfile(outDir, "trajectory"));

function writeSet(agent, O, prefix)
    % The compiled library takes single-precision input, so round the
    % observations to single first. Both sides then see identical inputs and
    % any remaining difference comes from the library, not from the cast.
    O = double(single(O));
    N = size(O, 1);
    A = zeros(N, getActionInfo(agent).Dimension(1));
    for i = 1:N
        act = getAction(agent, {O(i,:).'});
        A(i,:) = act{1}(:).';
    end
    writematrix(O, prefix + "_obs.csv");
    writematrix(A, prefix + "_action.csv");
    fprintf("%s: %d samples, obs %d-D, action %d-D, action range [%.3f, %.3f]\n", ...
        prefix, N, size(O,2), size(A,2), min(A(:)), max(A(:)));
end
