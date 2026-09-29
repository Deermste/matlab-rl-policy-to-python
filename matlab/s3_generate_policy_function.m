%% s3_generate_policy_function.m
% Turns the trained agent into a standalone MATLAB function:
%   build/evaluatePolicy.m   thin wrapper that runs the agent's policy
%   build/agentData.mat      serialized policy (weights, action limits, ...)
%
% generatePolicyFunction has no output-folder option, so the script changes
% into build/ first. Both files are also copied to artifacts/.

clear; clc;

root     = fileparts(fileparts(mfilename('fullpath')));
buildDir = fullfile(root, "build");
if ~isfolder(buildDir), mkdir(buildDir); end

S = load(fullfile(root, "artifacts", "agent.mat"));
agent = S.agent;
agent.UseExplorationPolicy = false;   % must be set BEFORE generating

oldDir = cd(buildDir);
generatePolicyFunction(agent);

assert(isfile("evaluatePolicy.m") && isfile("agentData.mat"), ...
    "generatePolicyFunction did not produce the expected files.");
copyfile("evaluatePolicy.m", fullfile(root, "artifacts"));
copyfile("agentData.mat",    fullfile(root, "artifacts"));

fprintf("\n=== evaluatePolicy.m ===\n");
type evaluatePolicy.m
cd(oldDir);   % back to where we started (scripts share the base workspace)
