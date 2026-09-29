%% s1_train_agent.m
% Trains a small PPO agent (Gaussian actor, continuous action) on the
% built-in CartPole-Continuous environment and saves it to
% artifacts/agent.mat.
%
% The trained agent is checked in, so you can skip this step and continue
% with s2_generate_validation_data.m. Set FORCE_RETRAIN = true to redo it.

clear; clc;
FORCE_RETRAIN = false;

here    = fileparts(mfilename('fullpath'));
outFile = fullfile(here, "..", "artifacts", "agent.mat");
if isfile(outFile) && ~FORCE_RETRAIN
    fprintf("%s exists, skipping training (set FORCE_RETRAIN = true to redo).\n", outFile);
    return
end

rng(0);   % reproducible weights and episodes

env     = rlPredefinedEnv("CartPole-Continuous");
obsInfo = getObservationInfo(env);   % 4x1: x, dx, theta, dtheta
actInfo = getActionInfo(env);        % 1x1 force, limited to [-10 10] N

% PPO agent with a small Gaussian actor and value critic (64 hidden units).
% The observation is not normalized (Normalization = "none" by default), the
% actor output is bounded to the action limits by a tanh + scaling layer.
init  = rlAgentInitializationOptions(NumHiddenUnit = 64);
agent = rlPPOAgent(obsInfo, actInfo, init);

opts = agent.AgentOptions;
opts.SampleTime        = env.Ts;
opts.ExperienceHorizon = 1024;
opts.MiniBatchSize     = 256;
opts.NumEpoch          = 5;
opts.EntropyLossWeight = 0.005;
opts.ActorOptimizerOptions.LearnRate  = 3e-4;
opts.CriticOptimizerOptions.LearnRate = 1e-3;
agent.AgentOptions = opts;

% Return per episode is at most ~500 (500 steps, reward <= 1 per step);
% falling costs -50. Stop once the moving average is clearly balancing.
trainOpts = rlTrainingOptions( ...
    MaxEpisodes = 2000, ...
    MaxStepsPerEpisode = 500, ...
    StopTrainingCriteria = "AverageReward", ...
    StopTrainingValue = 450, ...
    ScoreAveragingWindowLength = 20, ...
    Verbose = false, ...
    Plots = "none");

fprintf("Training PPO agent on CartPole-Continuous...\n");
tic
result = train(agent, env, trainOpts);
fprintf("Done after %d episodes (%.0f s), average reward %.1f\n", ...
    numel(result.EpisodeReward), toc, result.AverageReward(end));

% ---- add observation limits to the trained agent ---------------------------
% getAction clips the observation to these limits BEFORE the network sees it
% (the "ObservationBounder" in the generated code). That clipping is part of
% the policy but not part of the actor network, so a plain ONNX export of the
% network loses it (see s5 and docs/why-not-onnx.md).
% The limits are added after training because train() requires the agent's
% observation spec to match the environment's, and the predefined environment
% does not allow changing its spec. The network weights stay untouched.
limSpec = rlNumericSpec(obsInfo.Dimension, ...
    LowerLimit = [-2.4; -2; -0.21; -2], UpperLimit = [2.4; 2; 0.21; 2]);
limSpec.Name = obsInfo.Name;
limSpec.Description = obsInfo.Description;

actorNet  = getModel(getActor(agent));
criticNet = getModel(getCritic(agent));
actor  = rlContinuousGaussianActor(actorNet, limSpec, actInfo, ...
    ActionMeanOutputNames = actorNet.OutputNames{1}, ...
    ActionStandardDeviationOutputNames = actorNet.OutputNames{2}, ...
    ObservationInputNames = actorNet.InputNames{1});
critic = rlValueFunction(criticNet, limSpec);
agent  = rlPPOAgent(actor, critic, agent.AgentOptions);

% Deterministic evaluation of the agent as deployed (with limits).
agent.UseExplorationPolicy = false;
[~, ret] = rollout_cartpole(agent, env, 0, 500);
fprintf("Deterministic rollout return: %.0f / 500\n", ret);
assert(ret > 400, "Agent did not learn to balance. Re-run with another seed.");

if ~isfolder(fileparts(outFile)), mkdir(fileparts(outFile)); end
save(outFile, "agent");
fprintf("Saved %s\n", outFile);
