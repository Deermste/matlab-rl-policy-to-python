%% s4_build_policy_library.m
% Compiles build/evaluatePolicy.m (+ agentData.mat) with MATLAB Coder into a
% self-contained shared library and copies it to artifacts/<platform>/.
%
% Requirements: MATLAB Coder, Deep Learning Toolbox, Reinforcement Learning
% Toolbox, MATLAB Coder Interface for Deep Learning, a C++ compiler.

clear; clc;

root     = fileparts(fileparts(mfilename('fullpath')));
buildDir = fullfile(root, "build");
assert(isfile(fullfile(buildDir, "evaluatePolicy.m")), "Run s3_generate_policy_function first.");

S = load(fullfile(root, "artifacts", "agent.mat"));
obsDim = getObservationInfo(S.agent).Dimension(1);

cfg = coder.config('dll');
cfg.TargetLang          = 'C++';        % required for dlnetwork-based policies
cfg.GenCodeOnly         = false;        % build the library, not only the code
cfg.BuildConfiguration  = 'Faster Runs';
cfg.GenerateReport      = true;
cfg.GenerateExampleMain = 'DoNotGenerate';
cfg.EnableOpenMP        = false;        % avoids a runtime dependency on libiomp5

% No external deep learning backend (MKL-DNN, cuDNN): weights are baked into
% the generated code and the library stays self-contained for small networks.
cfg.DeepLearningConfig  = coder.DeepLearningConfig('TargetLibrary', 'none');

% cfg.Hardware is deliberately not set: Coder targets the host platform
% (.dll on Windows, .so on Linux / MATLAB Online).
if ispc
    platform = "win64";  libFile = "evaluatePolicy.dll";
else
    platform = "linux64"; libFile = "evaluatePolicy.so";
end

% Input is single precision, obsDim x 1.
obsArg = coder.typeof(single(0), [obsDim 1]);

oldDir = cd(buildDir);
fprintf("Running codegen for %s (this can take a few minutes)...\n", platform);
codegen evaluatePolicy -config cfg -args {obsArg} -report

cd(oldDir);
outDir = fullfile(buildDir, "codegen", "dll", "evaluatePolicy");
assert(isfile(fullfile(outDir, libFile)), "Expected %s was not built.", libFile);

destDir = fullfile(root, "artifacts", platform);
if ~isfolder(destDir), mkdir(destDir); end
copyfile(fullfile(outDir, libFile), destDir);
copyfile(fullfile(outDir, "evaluatePolicy.h"), fullfile(root, "artifacts"));

% ---- policy_meta.json: everything the Python wrapper needs to know ---------
actDim = getActionInfo(S.agent).Dimension(1);
hdr    = fileread(fullfile(outDir, "evaluatePolicy.h"));
sig    = regexp(hdr, "EVALUATEPOLICY_DLL_EXPORT extern[^;]*;", "match", "once");
meta = struct( ...
    "matlab_release",         string(matlabRelease.Release), ...
    "agent_class",            string(class(S.agent)), ...
    "obs_dim",                obsDim, ...
    "act_dim",                actDim, ...
    "obs_dtype",              "float32", ...
    "act_dtype",              "float64", ...
    "action_is_return_value", actDim == 1, ...   % Coder returns scalars by value
    "signature",              strtrim(regexprep(sig, "\s+", " ")), ...
    "library",                libFile);
fid = fopen(fullfile(root, "artifacts", "policy_meta.json"), "w");
fwrite(fid, jsonencode(meta, PrettyPrint = true));
fclose(fid);

fprintf("\n=== Built artifacts ===\n");
files = dir(outDir);
for k = 1:numel(files)
    if ~files(k).isdir, fprintf("  %s  (%d bytes)\n", files(k).name, files(k).bytes); end
end

fprintf("\n=== evaluatePolicy.h: check the signature ===\n");
type(fullfile(outDir, "evaluatePolicy.h"));
