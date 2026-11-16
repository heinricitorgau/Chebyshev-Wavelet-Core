function run_ci_checks()
%RUN_CI_CHECKS 持續整合（GitHub Actions）執行的 MATLAB 檢查；本機亦可直接呼叫
%
%   依序執行三項檢查，任一失敗即以 error 結束（CI 因此標示為失敗）：
%
%     1. checkcode 靜態檢查：所有 .m 檔皆須無任何訊息。
%     2. 核心數學：build_chebyshev_matrices(3, 4) 與論文 Eq.(4.9) 逐項誤差
%        須為 0，正交歸一性與積分精確性須達機器精度。
%     3. 雙軌一致性：verify_lean_agreement 須回報「全數相符」，即 Lean
%        已證定理的閉式結果與 MATLAB 實際建構的矩陣逐項一致（含 POM）。
%
%   刻意不執行 demo_omi_pom：它需要連網抓資料且需數分鐘，並會覆寫
%   figures/ 下的 PNG。CI 只檢查可離線、可決定性重現的性質。
%
%   語法：
%     run_ci_checks
%
%   See also BUILD_CHEBYSHEV_MATRICES, VERIFY_LEAN_AGREEMENT, CHECKCODE.
%
%   Author : Kao, En-Tsai
%   License: MIT (see LICENSE)

rootDir = setup_paths();

% ---------------------------------------------------------------------
% 1. checkcode：路徑一律由 project_root 推得，與目前工作目錄無關
% ---------------------------------------------------------------------
files = dir(fullfile(rootDir, '**', '*.m'));
nIssues = 0;
for i = 1:numel(files)
    f = fullfile(files(i).folder, files(i).name);
    r = checkcode(f);
    rel = erase(f, [rootDir filesep]);
    fprintf('checkcode %-45s %d issue(s)\n', rel, numel(r));
    for j = 1:numel(r)
        fprintf('    line %d: %s\n', r(j).line, r(j).message);
    end
    nIssues = nIssues + numel(r);
end
if nIssues > 0
    error('run_ci_checks:checkcode', 'checkcode reported %d issue(s).', nIssues);
end

% ---------------------------------------------------------------------
% 2. 核心數學：與論文 Eq.(4.9) 逐項比對（k = 3, M = 4）
% ---------------------------------------------------------------------
[~, ~, info] = build_chebyshev_matrices(3, 4, false, [], 'Verify', true);
v = info.verify;
fprintf('\nEq.(4.9) max |P - P_paper|     = %.3e\n', v.paperEq49);
fprintf('orthonormality max error       = %.3e\n', v.orthonormality);
fprintf('integration exactness max error = %.3e\n', v.integrationExactness);
if v.paperEq49 ~= 0
    error('run_ci_checks:eq49', ...
        'OMI differs from the paper''s Eq.(4.9) by %.3e (must be exactly 0).', v.paperEq49);
end
if ~(v.orthonormality <= 1e-12)
    error('run_ci_checks:ortho', 'Orthonormality error %.3e exceeds 1e-12.', v.orthonormality);
end
if ~(v.integrationExactness <= 1e-12)
    error('run_ci_checks:integration', ...
        'Integration exactness error %.3e exceeds 1e-12.', v.integrationExactness);
end

% ---------------------------------------------------------------------
% 3. 雙軌一致性：Lean 已證定理 vs MATLAB 實作
% ---------------------------------------------------------------------
fprintf('\n');
results = verify_lean_agreement();
if ~results.pass
    error('run_ci_checks:lean', ...
        'verify_lean_agreement reports a mismatch between the Lean theorems and the MATLAB matrices.');
end

fprintf('\nAll CI checks passed.\n');
end
