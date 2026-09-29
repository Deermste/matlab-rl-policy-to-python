%% check_requirements.m
% Verifies that everything needed for the export workflow is available.
% Run this first. It errors out with a list of what is missing.

clear; clc;
fprintf("MATLAB %s (%s)\n\n", version, computer);

% product name (as in ver), license feature name
required = {
    "Reinforcement Learning Toolbox", "Reinforcement_Learn_Toolbox"
    "Deep Learning Toolbox",          "Neural_Network_Toolbox"
    "MATLAB Coder",                   "MATLAB_Coder"
};

missing = strings(0,1);
for k = 1:size(required,1)
    ok = logical(license('test', required{k,2}));
    report(required{k,1}, ok);
    if ~ok, missing(end+1,1) = required{k,1}; end %#ok<SAGROW>
end

% "MATLAB Coder Interface for Deep Learning" is a support package. Its
% presence is detected by asking for the DL codegen config it provides.
try
    coder.DeepLearningConfig('TargetLibrary', 'none');
    ok = true;
catch
    ok = false;
end
report("MATLAB Coder Interface for Deep Learning", ok);
if ~ok, missing(end+1,1) = "MATLAB Coder Interface for Deep Learning (Add-On Explorer)"; end

% A C++ compiler is needed to build the shared library.
cc = mex.getCompilerConfigurations('C++', 'Selected');
ok = ~isempty(cc);
report("C++ compiler (mex -setup C++)", ok);
if ok, fprintf("    -> %s\n", cc(1).Name); else, missing(end+1,1) = "C++ compiler (run: mex -setup C++)"; end

fprintf("\n");
if isempty(missing)
    fprintf("All requirements satisfied.\n");
else
    error("Missing requirements:\n  - %s", strjoin(missing, newline + "  - "));
end

function report(name, ok)
    if ok, tag = "OK     "; else, tag = "MISSING"; end
    fprintf("  [%s] %s\n", tag, name);
end
