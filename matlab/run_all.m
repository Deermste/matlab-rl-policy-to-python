%% run_all.m
% Runs the whole export pipeline for the CartPole example, in order.
%
%   check_requirements            toolboxes, Coder add-on, C++ compiler
%   s1_train_agent                trains the agent            (skipped if artifacts/agent.mat exists)
%   s2_generate_validation_data   MATLAB getAction reference data
%   s3_generate_policy_function   generatePolicyFunction -> evaluatePolicy.m, agentData.mat
%   s4_build_policy_library       MATLAB Coder -> evaluatePolicy.dll / .so for THIS platform
%   s5_onnx_diagnostic            ONNX export of the actor network, for the comparison
%
% Optional, to build the Linux library on a machine without Linux MATLAB:
%   s6_generate_linux_sources     then: bash scripts/build_linux_so.sh
%
% Afterwards validate in Python:   python -m matlab_policy.validate
%
% Each script starts with `clear`, so they are called one after another
% instead of from a loop.

check_requirements
s1_train_agent
s2_generate_validation_data
s3_generate_policy_function
s4_build_policy_library
s5_onnx_diagnostic
fprintf("\nPipeline finished. Validate with:  python -m matlab_policy.validate\n");
