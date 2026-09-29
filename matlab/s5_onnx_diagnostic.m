%% s5_onnx_diagnostic.m
% Shows what exporting only the actor network to ONNX captures, compared to the
% full policy that getAction evaluates.
%
% Writes:
%   artifacts/onnx/policy.onnx              actor dlnetwork exported with exportONNXNetwork
%   artifacts/onnx/predict_net_outputs.csv  predict(net, obs) on the box observations
%                                           columns: mean (actDim), std (actDim)
%
% Then prints the three comparisons from docs/why-not-onnx.md. The ONNX vs
% MATLAB comparisons are repeated in Python by python/examples/02_onnx_vs_codegen.py.
%
% Requires the "Deep Learning Toolbox Converter for ONNX Model Format" add-on.

clear; clc;

root   = fileparts(fileparts(mfilename('fullpath')));
outDir = fullfile(root, "artifacts", "onnx");
if ~isfolder(outDir), mkdir(outDir); end

S = load(fullfile(root, "artifacts", "agent.mat"));
agent = S.agent;
agent.UseExplorationPolicy = false;

actor  = getActor(agent);
net    = getModel(actor);
actDim = getActionInfo(agent).Dimension(1);
obsDim = getObservationInfo(agent).Dimension(1);

fprintf("Actor: %s, normalization = %s\n", class(actor), string(actor.Normalization));
disp(net.Layers)
fprintf("Network outputs: %s\n", strjoin(net.OutputNames, ", "));

exportONNXNetwork(net, fullfile(outDir, "policy.onnx"));

% ---- reference data from s2 ----------------------------------------------
O   = readmatrix(fullfile(root, "artifacts", "validation", "box_obs.csv"));
Aga = readmatrix(fullfile(root, "artifacts", "validation", "box_action.csv"));   % getAction
assert(size(O,2) == obsDim);

% predict(net, obs): what the network alone computes.
[mu, sigma] = predict(net, dlarray(single(O.'), "CB"));
mu    = double(extractdata(mu)).';
sigma = double(extractdata(sigma)).';
writematrix([mu sigma], fullfile(outDir, "predict_net_outputs.csv"));

% The same network, but fed the observation clipped to the agent's limits.
lo = getObservationInfo(agent).LowerLimit.';
hi = getObservationInfo(agent).UpperLimit.';
Oc = min(max(O, lo), hi);
muClipped = double(extractdata(predict(net, dlarray(single(Oc.'), "CB")))).';
nClipped  = nnz(any(O < lo | O > hi, 2));

fprintf("\nOn %d box observations:\n", size(O,1));
fprintf("  predict(net) vs getAction   max|diff| = %.3e   (is the transform inside the net?)\n", ...
    max(abs(mu - Aga), [], "all"));
fprintf("  predict(net) on clipped obs vs getAction   max|diff| = %.3e   (%d of %d samples were clipped)\n", ...
    max(abs(muClipped - Aga), [], "all"), nClipped, size(O,1));
fprintf("  action range  getAction [%.3f, %.3f]   net mean [%.3f, %.3f]\n", ...
    min(Aga,[],"all"), max(Aga,[],"all"), min(mu,[],"all"), max(mu,[],"all"));
fprintf("  network std output range [%.3f, %.3f]   (not used by the deterministic policy)\n", ...
    min(sigma,[],"all"), max(sigma,[],"all"));
fprintf("\nNext: python python/examples/02_onnx_vs_codegen.py\n");
