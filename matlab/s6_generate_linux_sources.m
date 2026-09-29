%% s6_generate_linux_sources.m
% Generates the C++ sources of the policy library without building them, so a
% Linux .so can be compiled with g++ on a machine that has no MATLAB
% (scripts/build_linux_so.sh, e.g. inside WSL).
%
% Alternative to running s4_build_policy_library.m on a Linux MATLAB or on
% MATLAB Online. The generated code with TargetLibrary = 'none' is plain C++
% without MATLAB runtime dependencies.
%
% Output: build/linux_src/*.cpp, *.h

clear; clc;

root     = fileparts(fileparts(mfilename('fullpath')));
buildDir = fullfile(root, "build", "linux_gen");
srcDir   = fullfile(root, "build", "linux_src");
assert(isfile(fullfile(root, "build", "evaluatePolicy.m")), "Run s3_generate_policy_function first.");

S = load(fullfile(root, "artifacts", "agent.mat"));
obsDim = getObservationInfo(S.agent).Dimension(1);

if isfolder(buildDir), rmdir(buildDir, "s"); end
if isfolder(srcDir),   rmdir(srcDir,   "s"); end
mkdir(buildDir); mkdir(srcDir);
copyfile(fullfile(root, "build", "evaluatePolicy.m"), buildDir);
copyfile(fullfile(root, "build", "agentData.mat"),    buildDir);

cfg = coder.config('dll');
cfg.TargetLang          = 'C++';
cfg.GenCodeOnly         = true;         % generate sources, do not compile
cfg.BuildConfiguration  = 'Faster Runs';
cfg.GenerateReport      = false;
cfg.GenerateExampleMain = 'DoNotGenerate';
cfg.EnableOpenMP        = false;
cfg.DeepLearningConfig  = coder.DeepLearningConfig('TargetLibrary', 'none');

oldDir = cd(buildDir);
codegen evaluatePolicy -config cfg -args {coder.typeof(single(0), [obsDim 1])}
cd(oldDir);

gen = fullfile(buildDir, "codegen", "dll", "evaluatePolicy");
for pat = ["*.cpp", "*.h"]
    files = dir(fullfile(gen, pat));
    for k = 1:numel(files)
        copyfile(fullfile(gen, files(k).name), srcDir);
    end
end
% The generated rtwtypes.h includes MATLAB's tmwtypes.h. It is copied next to
% the sources only for the local build; build/ is not committed.
copyfile(fullfile(matlabroot, "extern", "include", "tmwtypes.h"), srcDir);
n = numel(dir(fullfile(srcDir, "*.cpp")));
fprintf("\n%d C++ source files written to %s\n", n, srcDir);
fprintf("Next (Linux / WSL): bash scripts/build_linux_so.sh\n");
