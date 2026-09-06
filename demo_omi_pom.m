%% DEMO_OMI_POM  第二類 Chebyshev 小波運算矩陣示範腳本
%
%   本腳本示範本套件的完整用法，共十一個章節：
%
%     1. 建構 OMI 並與論文 Eq.(4.9) 逐項比對
%     2. 小波基底函數視覺化
%     3. 函數逼近與收斂階數驗證 (論文 Thm 3.2)
%     4. 乘積運算矩陣 POM 的作用
%     5. 論文 Example 1：一階線性 ODE
%     6. 變係數 ODE：POM 的實際應用
%     7. 效能與 GPU 加速
%     8. 應用：金融時間序列去噪與趨勢特徵 (wavelet_denoise_series)
%     9. 因果特徵萃取與前視偏誤量化 (wavelet_features)
%    10. 預測模型與 walk-forward 回測 (walkforward_backtest)
%    11. 橫斷面多空策略 (cross_sectional_backtest)
%
%   於 MATLAB 編輯器中可用 Ctrl+Enter 逐節執行；亦可直接
%       >> demo_omi_pom
%   完整執行。
%
%   參考文獻：
%     H. K. Nigam and M. M. Alam, Convergence analysis of an efficient
%     Chebyshev wavelet and its applications to differential equations via
%     operational matrices of integration, Tamkang J. Math. 57(3),
%     171-193, 2026. DOI:10.5556/j.tkjm.57.2026.5958
%
%   Author : Kao, En-Tsai
%   License: MIT (see LICENSE)

clear; clc; close all;
SHOW_PLOTS = true;      % 設為 false 可在無圖形環境 (如 CI) 下執行
EXPORT_PNG = false;     % 設為 true 可將圖表輸出成 PNG (README 所用圖檔)
FIG_DIR    = 'figures'; % PNG 輸出目錄
%
% 圖表文字一律使用英文：MATLAB 預設字型不含 CJK 字元，中文標籤在
% exportgraphics 輸出的 PNG 中會顯示為方框，且其他平台未必安裝中文字型。
% (FIG_DIR 於實際輸出時才建立，見檔案末端的 export_png)

fprintf('===============================================================\n');
fprintf(' 第二類 Chebyshev 小波：OMI 與 POM 示範\n');
fprintf(' MATLAB %s\n', version);
fprintf('===============================================================\n');


%% 1. 建構 OMI 並驗證 =====================================================
% 取論文 Sec.4.1 之參數 k = 3, M = 4，矩陣階數 N = 2^(k-1)*M = 16。
fprintf('\n【1】建構 OMI (k = 3, M = 4) 並與論文 Eq.(4.9) 比對\n');

k = 3;  M = 4;
[P, ~, info] = build_chebyshev_matrices(k, M, false, [], 'Verify', true);

fprintf('  矩陣階數 N = %d，建構耗時 %.4f 秒\n', info.N, info.timeOMI);
fprintf('  與論文 Eq.(4.9) 之最大逐項差異 : %.3e\n', info.verify.paperEq49);
fprintf('  正交歸一性誤差                 : %.3e\n', info.verify.orthonormality);
fprintf('  積分精確性 (m <= M-2 之列)     : %.3e\n', info.verify.integrationExactness);

% 對角區塊 M 與離對角區塊 N (論文 Lemma 4.1)
fprintf('\n  對角區塊 M (乘上 2^k = %d 後之整數/分數形式)：\n', 2^k);
disp(info.Mblk * 2^k);
fprintf('  離對角區塊 N (僅第一行非零)：\n');
disp(info.Nblk * 2^k);

% (P 與 POM 的非零元分布圖統一於第 4 節繪製)


%% 2. 小波基底函數視覺化 ==================================================
% \psi_{n,m}(t) 於第 n 個子區間上為 m 次多項式，其餘處為 0。
fprintf('\n【2】繪製 16 個基底函數 \\psi_{n,m}(t)\n');

if SHOW_PLOTS
    tt  = linspace(0, 1, 4001);
    Psi = info.basis(tt);

    figBasis = figure('Name', 'Basis functions', 'Color', 'w', ...
        'Position', [100 100 900 640]);
    for m = 0:M-1
        subplot(M, 1, m+1); hold on; grid on; box on;
        for n = 1:info.L
            idx  = (n-1)*M + m + 1;
            mask = tt >= (n-1)/info.L & tt < n/info.L;
            plot(tt(mask), Psi(idx, mask), 'LineWidth', 1.6);
        end
        for b = 1:info.L-1                      % 子區間分界線
            xline(b/info.L, ':', 'Color', [.6 .6 .6]);
        end
        ylabel(sprintf('$m = %d$', m), 'Interpreter', 'latex', 'FontSize', 12);
        xlim([0 1]); set(gca, 'FontSize', 10);
        if m == 0
            title('Second-kind Chebyshev wavelets $\psi_{n,m}(t)$, $k=3$, $M=4$', ...
                'Interpreter', 'latex', 'FontSize', 13);
        end
    end
    xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 12);
    export_png(figBasis, 'fig_basis', EXPORT_PNG, FIG_DIR);
end
fprintf('  基底函數在各子區間 [%g, %g), ... 上分段定義\n', 0, 1/info.L);


%% 3. 函數逼近與收斂階數 ==================================================
% 論文 Thm 3.2：E_{2^{k-1},J} = O( 1 / (J! 2^{J(k+1)}) )。
% 固定 M，逐次加密解析度 k，觀察 L^2 誤差的收斂速率。
fprintf('\n【3】函數逼近收斂性 (f(t) = exp(t)sin(4t)，固定 M = %d)\n', 4);

f      = @(t) exp(t) .* sin(4*t);
tt     = linspace(0, 1, 20001).';
Mfix   = 4;
kList  = 1:7;
errL2  = zeros(numel(kList), 1);

for ii = 1:numel(kList)
    [~, ~, ik] = build_chebyshev_matrices(kList(ii), Mfix);
    Ck         = ik.expand(f);
    fApprox    = (Ck.' * ik.basis(tt)).';
    errL2(ii)  = sqrt(trapz(tt, (fApprox - f(tt)).^2));
end

rate = [NaN; log2(errL2(1:end-1) ./ errL2(2:end))];
fprintf('   k    N      L2 誤差      收斂階 (log2 比值)\n');
for ii = 1:numel(kList)
    fprintf('  %2d  %4d   %10.3e   %s\n', kList(ii), 2^(kList(ii)-1)*Mfix, ...
        errL2(ii), ternary(isnan(rate(ii)), '   --', sprintf('%7.2f', rate(ii))));
end
fprintf('  理論上每提高一階 k 應約降低 2^M = %d 倍 (M = %d)\n', 2^Mfix, Mfix);

if SHOW_PLOTS
    figConv = figure('Name', 'Convergence', 'Color', 'w', ...
        'Position', [100 100 760 560]);
    Nv = 2.^(kList-1)*Mfix;
    loglog(Nv, errL2, 'o-', 'LineWidth', 1.8, 'MarkerSize', 8, ...
        'MarkerFaceColor', 'w'); hold on; grid on; box on;
    ref = errL2(1) * (Nv/Nv(1)).^(-Mfix);       % 理論斜率 -M
    loglog(Nv, ref, '--', 'Color', [.5 .5 .5], 'LineWidth', 1.4);
    legend({'measured $L^2$ error', ...
            sprintf('reference slope $\\hat{N}^{-%d}$', Mfix)}, ...
        'Interpreter', 'latex', 'FontSize', 11, 'Location', 'southwest');
    xlabel('$\hat{N} = 2^{k-1}M$', 'Interpreter', 'latex', 'FontSize', 13);
    ylabel('$\|f - f_{\mathrm{approx}}\|_{L^2}$', 'Interpreter', 'latex', ...
        'FontSize', 13);
    title({'Convergence of the wavelet approximation', ...
           '$f(t) = e^{t}\sin(4t)$, fixed $M = 4$'}, ...
        'Interpreter', 'latex', 'FontSize', 13);
    set(gca, 'FontSize', 11);
    export_png(figConv, 'fig_convergence', EXPORT_PNG, FIG_DIR);
end


%% 4. 乘積運算矩陣 POM ====================================================
% POM 滿足 F^T \Psi(t)\Psi^T(t) = \Psi^T(t)\tilde{F}，用於處理兩函數相乘。
fprintf('\n【4】乘積運算矩陣 POM 的結構與精度\n');

kp = 4;  Mp = 6;
[~, ~, ip] = build_chebyshev_matrices(kp, Mp);
A  = ip.expand(@(t) t);                       % a(t) = t
[~, Atil, ipv] = build_chebyshev_matrices(kp, Mp, false, A, 'Verify', true);

fprintf('  k = %d, M = %d -> \\tilde{F} 為 %d 個 %dx%d 區塊組成的區塊對角矩陣\n', ...
    kp, Mp, ipv.L, Mp, Mp);
fprintf('  POM 投影一致性誤差 : %.3e\n', ipv.verify.pomProjection);
fprintf('  POM 區塊對稱性誤差 : %.3e\n', ipv.verify.pomSymmetry);

% 以 a(t)*b(t) 檢視截斷效應。關鍵在於乘積的「局部多項式次數」：
%   deg(a)+deg(b) <= M-1 -> 乘積仍在基底空間內，逼近為精確
%   deg(a)+deg(b) >  M-1 -> 高次成分被投影截斷，出現逼近誤差
tq = linspace(0, 1, 2001).';
Pq = ip.basis(tq);
fprintf('\n  a(t) = t (局部次數 1)，M = %d (可表示至 %d 次)：\n', Mp, Mp-1);
for deg = [2 5]
    B  = ip.expand(@(t) t.^deg);
    prodApprox = (Atil * B).' * Pq;            % a*b 的小波係數 = \tilde{A} B
    prodExact  = (tq.^(deg+1)).';
    tag = '截斷 -> 有誤差';
    if 1 + deg <= Mp - 1
        tag = '未截斷 -> 精確';
    end
    fprintf('    b(t) = t^%d : deg(a*b) = %d, 最大誤差 %.3e  (%s)\n', ...
        deg, deg+1, max(abs(prodApprox - prodExact)), tag);
end

if SHOW_PLOTS
    figStruct = figure('Name', 'Matrix structure', 'Color', 'w', ...
        'Position', [100 100 900 440]);
    subplot(1,2,1);
    spy(sparse(P), 8);                          % 第 1 節建構之 OMI (k=3, M=4)
    title({'OMI $P$ : block upper triangular', '$k=3$, $M=4$, $\hat{N}=16$'}, ...
        'Interpreter', 'latex', 'FontSize', 12);
    xlabel(sprintf('nnz = %d (%.1f%%)', nnz(P), 100*nnz(P)/numel(P)), ...
        'FontSize', 10);
    ylabel('row'); set(gca, 'FontSize', 10);

    subplot(1,2,2);
    spy(sparse(Atil), 6);
    title({'POM $\tilde{F}$ : block diagonal', ...
           sprintf('$k=%d$, $M=%d$, $\\hat{N}=%d$', kp, Mp, ipv.N)}, ...
        'Interpreter', 'latex', 'FontSize', 12);
    xlabel(sprintf('nnz = %d (%.1f%%)', nnz(Atil), 100*nnz(Atil)/numel(Atil)), ...
        'FontSize', 10);
    ylabel('row'); set(gca, 'FontSize', 10);
    export_png(figStruct, 'fig_structure', EXPORT_PNG, FIG_DIR);
end


%% 5. 論文 Example 1：一階線性 ODE ========================================
%   y'(t) + 2y(t) = t,  y(0) = 0,   精確解 y = t/2 - 1/4 + e^{-2t}/4
%
%   令 y'(t) = C^T\Psi(t)，兩側自 0 積分並代入 y(0)=0 與 t = E^T\Psi(t)：
%       C^T\Psi + 2C^T P\Psi = E^T P\Psi  =>  (I + 2P^T)C = P^T E
fprintf('\n【5】論文 Example 1： y'' + 2y = t,  y(0) = 0\n');

[P1, ~, i1] = build_chebyshev_matrices(3, 4);
E   = i1.expand(@(t) t);                       % 論文 Eq.(5.5)
C1  = (eye(i1.N) + 2*P1.') \ (P1.'*E);         % 論文 Eq.(5.7)
y1  = @(t) (C1.' * i1.basis(t)).';

fprintf('  c_{1,0} = %.15f\n', C1(1));
fprintf('  論文值  = 0.003866313244751 (Sec.5, Example 1)\n');

tt   = linspace(0, 1, 501).';
yExa = tt/2 - 1/4 + exp(-2*tt)/4;
fprintf('  最大絕對誤差 (k=3, M=4) : %.3e\n', max(abs(y1(tt) - yExa)));

% 提高解析度後的誤差
for kk = [4 5 6]
    [Pk, ~, ik] = build_chebyshev_matrices(kk, 4);
    Ek = ik.expand(@(t) t);
    Ck = (eye(ik.N) + 2*Pk.') \ (Pk.'*Ek);
    yk = (Ck.' * ik.basis(tt)).';
    fprintf('  最大絕對誤差 (k=%d, M=4) : %.3e\n', kk, max(abs(yk - yExa)));
end

if SHOW_PLOTS
    figEx1 = figure('Name', 'Example 1', 'Color', 'w', ...
        'Position', [100 100 820 620]);
    subplot(2,1,1);
    plot(tt, yExa, 'k-', 'LineWidth', 2.4); hold on; grid on; box on;
    plot(tt, y1(tt), '--', 'Color', [.85 .16 .16], 'LineWidth', 1.8);
    legend({'exact solution', 'wavelet solution ($k=3$, $M=4$)'}, ...
        'Interpreter', 'latex', 'FontSize', 11, 'Location', 'northwest');
    ylabel('$y(t)$', 'Interpreter', 'latex', 'FontSize', 13);
    title(['Example 1 (paper Sec.5): $y'' + 2y = t$, $y(0)=0$, ' ...
           'exact $y = \frac{t}{2}-\frac14+\frac14e^{-2t}$'], ...
        'Interpreter', 'latex', 'FontSize', 13);
    set(gca, 'FontSize', 11);

    subplot(2,1,2); hold on; grid on; box on;
    cmap = lines(4);
    for jj = 1:4
        kk = jj + 2;                            % k = 3,4,5,6
        [Pk, ~, ik] = build_chebyshev_matrices(kk, 4);
        Ek = ik.expand(@(t) t);
        Ck = (eye(ik.N) + 2*Pk.') \ (Pk.'*Ek);
        semilogy(tt, abs((Ck.'*ik.basis(tt)).' - yExa), 'LineWidth', 1.5, ...
            'Color', cmap(jj,:), 'DisplayName', sprintf('$k = %d$', kk));
    end
    set(gca, 'YScale', 'log', 'FontSize', 11);
    legend('Interpreter', 'latex', 'FontSize', 10, 'Location', 'east', ...
        'NumColumns', 2);
    xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 13);
    ylabel('absolute error', 'FontSize', 12);
    title('Error decay with refinement ($M = 4$ fixed)', ...
        'Interpreter', 'latex', 'FontSize', 12);
    export_png(figEx1, 'fig_example1', EXPORT_PNG, FIG_DIR);
end


%% 6. 變係數 ODE：POM 的實際應用 ==========================================
%   y'(t) + t\,y(t) = t,  y(0) = 0,   精確解 y = 1 - e^{-t^2/2}
%
%   令 y' = C^T\Psi，則 y = (P^T C)^T\Psi =: D^T\Psi。變係數項用 POM 處理：
%       a(t)y(t) = (A^T\Psi)(\Psi^T D) = \Psi^T \tilde{A} D,
%   代入方程並比較係數，得線性系統
%       (I + \tilde{A}P^T)\,C = G.
fprintf('\n【6】變係數 ODE： y'' + t*y = t,  y(0) = 0  (需用到 POM)\n');

for kk = [3 5 7]
    Mv = 6;
    [Pv, ~, iv] = build_chebyshev_matrices(kk, Mv);
    Av = iv.expand(@(t) t);                    % a(t) = t
    Gv = iv.expand(@(t) t);                    % g(t) = t
    [~, Atl] = build_chebyshev_matrices(kk, Mv, false, Av);

    Cv = (eye(iv.N) + Atl*Pv.') \ Gv;          % 求解 y' 的係數
    Dv = Pv.' * Cv;                            % y 的係數 (y(0) = 0)
    yv = (Dv.' * iv.basis(tt)).';

    fprintf('  k = %d, M = %d : 最大絕對誤差 %.3e\n', kk, Mv, ...
        max(abs(yv - (1 - exp(-tt.^2/2)))));
    if kk == 7 && SHOW_PLOTS
        figVar = figure('Name', 'Variable-coefficient ODE', 'Color', 'w', ...
            'Position', [100 100 820 460]);
        plot(tt, 1 - exp(-tt.^2/2), 'k-', 'LineWidth', 2.4); hold on;
        grid on; box on;
        plot(tt, yv, '--', 'Color', [.85 .16 .16], 'LineWidth', 1.8);
        legend({'exact solution $y = 1-e^{-t^2/2}$', ...
                sprintf('wavelet solution ($k=%d$, $M=%d$)', kk, Mv)}, ...
            'Interpreter', 'latex', 'FontSize', 11, 'Location', 'northwest');
        xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 13);
        ylabel('$y(t)$', 'Interpreter', 'latex', 'FontSize', 13);
        title(sprintf(['Variable-coefficient ODE $y'' + t\\,y = t$, ' ...
                       '$y(0)=0$ (solved via POM), max error %.1e'], ...
                      max(abs(yv - (1 - exp(-tt.^2/2))))), ...
            'Interpreter', 'latex', 'FontSize', 13);
        set(gca, 'FontSize', 11);
        export_png(figVar, 'fig_varcoef', EXPORT_PNG, FIG_DIR);
    end
end
fprintf(['  註：本例 a(t) = t 的局部次數僅為 1，POM 截斷影響輕微，\n' ...
         '      故誤差可隨 k 加密迅速降至機器精度。若變係數本身為\n' ...
         '      高次多項式，截斷效應會明顯得多 (見第 4 節)。\n']);


%% 7. 效能與 GPU 加速 =====================================================
fprintf('\n【7】效能：稀疏格式與 GPU 加速\n');

% 稀疏格式：密度趨近 1/(4M)
[Ps, ~, is] = build_chebyshev_matrices(12, 8, false, [], 'Format', 'sparse');
fprintf('  sparse k=12, M=8 : N = %d, nnz = %d, 密度 = %.5f (理論 %.5f)\n', ...
    is.N, nnz(Ps), nnz(Ps)/is.N^2, 1/(4*8));
fprintf('                     記憶體 %.1f MB，建構 %.3f 秒\n', ...
    is.bytes/2^20, is.timeOMI);
clear Ps

% CPU / GPU 矩陣相乘：P * X，X 為稠密資料矩陣 (貼近實際用途)
%
%   量測方法說明：
%   * 以 timeit / gputimeit 取穩健中位數，而非單次 tic/toc。
%   * 先做 P = P + 0 強制實體化所有記憶體頁；MATLAB 的 zeros() 對未寫入
%     的頁面採惰性配置，直接量測 P*P 會得到不具代表性的結果。
%   * 即便如此，CPU 端的 BLAS 執行緒狀態在部分機器上仍會在不同 MATLAB
%     行程間波動 (本機實測可達數倍)，故「加速比」僅供參考；GPU 端的
%     吞吐量則相當穩定。請以自己硬體上的實測結果為準。
kg = 9;  Mg = 8;                               % N = 2048，控制示範耗時
hasGPU = exist('canUseGPU', 'file') == 2 && canUseGPU();

for pc = {'double', 'single'}
    p = pc{1};
    [Pc, ~, ic] = build_chebyshev_matrices(kg, Mg, false, [], 'Precision', p);
    Pc = Pc + 0;                               % 實體化所有頁面
    Nc = ic.N;
    X  = rand(Nc, Nc, p);
    Yc = Pc * X;                               % 暖機
    tCPU = timeit(@() Pc*X, 1);
    fprintf('  [%-6s N=%d] CPU : %7.4f 秒 (%5.0f GFLOPS)', ...
        p, Nc, tCPU, 2*Nc^3/tCPU/1e9);
    if hasGPU
        Pg = gpuArray(Pc);  Xg = gpuArray(X);
        Yg = Pg * Xg;  wait(gpuDevice);        % 暖機
        tGPU = gputimeit(@() Pg*Xg);
        fprintf(' | GPU : %7.4f 秒 (%5.0f GFLOPS) | 比值 %.1fx | 差異 %.1e\n', ...
            tGPU, 2*Nc^3/tGPU/1e9, tCPU/tGPU, ...
            max(max(abs(double(gather(Yg)) - double(Yc)))));
        clear Pg Xg Yg
    else
        fprintf(' | (無可用 GPU)\n');
    end
    clear Pc X Yc
end
fprintf(['  註：消費級 GeForce 顯卡的 FP64 吞吐量僅為 FP32 的 1/64，\n' ...
         '      double 精度下 GPU 未必勝過 CPU；single 精度才能發揮效能。\n']);

%% 8. 應用：金融時間序列去噪與趨勢特徵 ====================================
% 以 wavelet_denoise_series 將含噪價格投影至小波空間、濾除高頻成分，
% 並由解析微分矩陣同步取得變化率。
fprintf('\n【8】金融時間序列去噪 (wavelet_denoise_series)\n');

rng(42);
nObs    = 750;
tDays   = (1:nObs).';
truePx  = 100 + 12*sin(2*pi*tDays/250) + cumsum(0.06*randn(nObs,1));  % sine + random walk
noisyPx = truePx + 1.2*randn(nObs,1);                                 % 加入高斯雜訊

[pxSmooth, pxSlope, ~, dnDiag] = wavelet_denoise_series(noisyPx, tDays, 4, 4);

fprintf('  雜訊估計 %.3f (真實 1.200) | 殘差 std %.3f | SNR 改善 %.1f dB\n', ...
    dnDiag.sigmaHat, dnDiag.residualStd, dnDiag.snrGainDb);
fprintf('  RMSE vs 真實訊號 : 原始 %.3f -> 去噪後 %.3f\n', ...
    sqrt(mean((noisyPx - truePx).^2)), sqrt(mean((pxSmooth - truePx).^2)));
fprintf('  子區間界點最大跳躍 : %.1f%% of std(S)\n', 100*dnDiag.maxBlockJumpRel);

% 參數掃描：殘差與界點跳躍需一併考量
fprintf('   k    每區塊點數   殘差std   界點跳躍   RMSE\n');
for kk = 3:7
    [ss, ~, ~, dd] = wavelet_denoise_series(noisyPx, tDays, kk, 4);
    fprintf('  %2d   %8.1f   %7.3f   %6.1f%%   %6.3f\n', kk, dd.pointsPerBlock, ...
        dd.residualStd, 100*dd.maxBlockJumpRel, sqrt(mean((ss - truePx).^2)));
end

if SHOW_PLOTS
    figDn = figure('Name', 'Wavelet denoising', 'Color', 'w', ...
        'Position', [80 80 980 720]);
    bnd = (1:info.L-1) * nObs / info.L;      % 以 k=4 的 8 個子區間為界

    subplot(3,1,1);
    plot(tDays, noisyPx, '-', 'Color', [.72 .72 .75], 'LineWidth', .8);
    hold on; grid on; box on;
    plot(tDays, truePx, 'k-', 'LineWidth', 1.8);
    plot(tDays, pxSmooth, '-', 'Color', [.85 .16 .16], 'LineWidth', 1.8);
    xline((1:7)*nObs/8, ':', 'Color', [.6 .6 .6]);
    legend({'noisy price', 'true signal', 'wavelet denoised'}, ...
        'Location', 'northwest', 'FontSize', 10);
    ylabel('price', 'FontSize', 12);
    title('Chebyshev wavelet denoising of a simulated price series ($k=4$, $M=4$)', ...
        'Interpreter', 'latex', 'FontSize', 13);
    set(gca, 'FontSize', 10);

    subplot(3,1,2);
    plot(tDays, noisyPx - pxSmooth, '-', 'Color', [.25 .45 .85], 'LineWidth', .8);
    grid on; box on;
    xline((1:7)*nObs/8, ':', 'Color', [.6 .6 .6]);
    ylabel('residual', 'FontSize', 12);
    title(sprintf(['Residual: std = %.3f vs true noise std = 1.200 ' ...
                   '(structureless $\\Rightarrow$ good fit)'], dnDiag.residualStd), ...
        'Interpreter', 'latex', 'FontSize', 12);
    set(gca, 'FontSize', 10);

    subplot(3,1,3);
    plot(tDays, pxSlope, '-', 'Color', [.1 .5 .3], 'LineWidth', 1.5);
    hold on; grid on; box on;
    plot(tDays, 12*(2*pi/250)*cos(2*pi*tDays/250), 'k--', 'LineWidth', 1.2);
    yline(0, '-', 'Color', [.4 .4 .4]);
    xline((1:7)*nObs/8, ':', 'Color', [.6 .6 .6]);
    legend({'$dS/dt$ (wavelet)', 'derivative of the sine component'}, ...
        'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 10);
    xlabel('$t$ (trading days)', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('$dS/dt$', 'Interpreter', 'latex', 'FontSize', 12);
    title(['Derivative as a trend-strength feature ' ...
           '(note the jumps at the dotted block boundaries)'], ...
        'Interpreter', 'latex', 'FontSize', 12);
    set(gca, 'FontSize', 10);
    export_png(figDn, 'fig_denoise', EXPORT_PNG, FIG_DIR);
end
fprintf(['  註：本基底於子區間界點不連續，導數在界點（圖中虛線）會出現\n' ...
         '      可見的跳躍，此為方法固有性質，詳見 help wavelet_denoise_series。\n']);


%% 9. 因果特徵與前視偏誤 ==================================================
% wavelet_features 以滾動視窗確保「時刻 t 的特徵只用到 t 之前的資料」。
% 本節量化：若改用第 8 節的批次去噪結果當特徵，會虛增多少樣本外準確率。
fprintf('\n【9】因果特徵萃取與前視偏誤量化 (wavelet_features)\n');

nSeed   = 20;
nS      = 1500;
tIdx    = (1:nS).';
splitAt = round(0.7 * nS);
accRW   = zeros(nSeed, 4);
accSig  = zeros(nSeed, 1);

% 本節刻意以「非因果」方式使用批次去噪作為反例，其界點跳躍警告在此無意義，
% 故暫時關閉；離開本節時還原。
wState = warning('off', 'wavelet_denoise_series:blockJump');
cleanupWarn = onCleanup(@() warning(wState));

for sd = 1:nSeed
    % (I) 純隨機漫步：不具可預測性，樣本外高於 0.5 即代表洩漏
    rng(1000 + sd);
    pw = 100 * cumprod(1 + 0.0004 + 0.011*randn(nS,1));
    yw = [sign(diff(pw)); NaN];
    [bSm, bSl] = wavelet_denoise_series(pw, tIdx, 4, 4);      % 批次（非因果）
    Fw = wavelet_features(pw, tIdx);                          % 因果
    cand = {Fw, bSl, bSm - pw, [bSl, bSm - pw]};
    for c = 1:4
        accRW(sd, c) = local_dir_acc(cand{c}, yw, splitAt);
    end

    % (II) 含真實週期訊號：因果特徵應抓得到
    rng(2000 + sd);
    ps = 100 + 15*sin(2*pi*(1:nS).'/180) + cumsum(0.25*randn(nS,1)) + randn(nS,1);
    accSig(sd) = local_dir_acc(wavelet_features(ps, tIdx), ...
                               [sign(diff(ps)); NaN], splitAt);
end

labels = {'因果特徵', '批次斜率', '批次平滑-價格', '兩者合用'};
fprintf('  純隨機漫步 (%d 次實驗) 的樣本外準確率：\n', nSeed);
for c = 1:4
    m  = mean(accRW(:,c));
    se = std(accRW(:,c)) / sqrt(nSeed);
    fprintf('    %-14s %.4f +- %.4f  (距 0.5 為 %+.1f 個標準誤)\n', ...
        labels{c}, m, se, (m - 0.5)/se);
end
fprintf('  含真實訊號序列，因果特徵：%.4f +- %.4f\n', ...
    mean(accSig), std(accSig)/sqrt(nSeed));
fprintf('  => 因果特徵在無訊號時落在 0.5（未捏造訊號），有訊號時抓得到。\n');

if SHOW_PLOTS
    rng(1001);
    pw = 100 * cumprod(1 + 0.0004 + 0.011*randn(nS,1));
    [bSm, ~] = wavelet_denoise_series(pw, tIdx, 4, 4);
    [~, ~, cz] = wavelet_features(pw, tIdx);

    figCz = figure('Name', 'Causal features', 'Color', 'w', ...
        'Position', [90 90 960 640]);
    subplot(2,1,1);
    seg = 400:700;
    plot(seg, pw(seg), '-', 'Color', [.72 .72 .75], 'LineWidth', .9);
    hold on; grid on; box on;
    plot(seg, bSm(seg), '-', 'Color', [.85 .16 .16], 'LineWidth', 1.8);
    plot(seg, cz.smooth(seg,2), '-', 'Color', [.1 .35 .75], 'LineWidth', 1.8);
    legend({'price', 'batch smoother (uses future data)', ...
            'causal smoother (past only, $W=63$)'}, ...
        'Interpreter', 'latex', 'Location', 'best', 'FontSize', 10);
    xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('price', 'FontSize', 12);
    title('Batch (non-causal) vs causal smoothing', ...
        'Interpreter', 'latex', 'FontSize', 13);
    set(gca, 'FontSize', 10);

    subplot(2,1,2);
    mv = mean(accRW, 1);
    sev = std(accRW, 0, 1) / sqrt(nSeed);
    b = bar(mv, 'FaceColor', 'flat');
    b.CData = [0.15 0.45 0.75; 0.85 0.45 0.16; 0.85 0.30 0.16; 0.75 0.12 0.12];
    hold on; grid on; box on;
    errorbar(1:4, mv, sev, 'k.', 'LineWidth', 1.2);
    yline(0.5, 'k--', 'LineWidth', 1.4);
    set(gca, 'XTickLabel', {'causal (ours)', 'batch slope', ...
                            'batch smooth - price', 'both'}, 'FontSize', 10);
    ylim([0.45 0.62]);
    ylabel('out-of-sample accuracy', 'FontSize', 12);
    title(['Directional accuracy on pure random walks ' ...
           '($n_{\mathrm{seed}} = ' num2str(nSeed) '$): ' ...
           'anything above the dashed line is leakage'], ...
        'Interpreter', 'latex', 'FontSize', 12);
    export_png(figCz, 'fig_causal', EXPORT_PNG, FIG_DIR);
end


%% 10. 預測模型與 walk-forward 回測 ========================================
% 先確認回測框架本身在「無訊號」資料上不會產生績效，再看它能否偵測到
% 真實訊號。順序不可顛倒：框架未經證實之前，任何績效數字都不可信。
fprintf('\n【10】預測模型與 walk-forward 回測 (walkforward_backtest)\n');

nB   = 2000;
tB   = (1:nB).';
nullR = 100;

% (I) 純隨機漫步：不具可預測性
rng(301);
pxRW = 100 * cumprod(1 + 0.0003 + 0.011*randn(nB,1));
Frw  = wavelet_features(pxRW, tB);
rRW  = walkforward_backtest(Frw, pxRW, 'NullRuns', nullR, 'CostBps', 5);

% (II) 含真實週期成分：可預測
rng(11);
pxSG = 100 + 15*sin(2*pi*(1:nB).'/180) + cumsum(0.25*randn(nB,1)) + randn(nB,1);
Fsg  = wavelet_features(pxSG, tB);
rSG  = walkforward_backtest(Fsg, pxSG, 'NullRuns', nullR, 'CostBps', 5);

fprintf('  %-16s %-10s %-12s %-10s %-12s %s\n', ...
    '資料', '準確率', 'p(準確率)', 'Sharpe', 'p(Sharpe)', '樣本外筆數');
fprintf('  %-16s %-10.4f %-12.4f %-10.2f %-12.4f %d\n', '純隨機漫步', ...
    rRW.accuracy, rRW.null.pAccuracy, rRW.sharpe, rRW.null.pSharpe, rRW.nTest);
fprintf('  %-16s %-10.4f %-12.4f %-10.2f %-12.4f %d\n', '含週期訊號', ...
    rSG.accuracy, rSG.null.pAccuracy, rSG.sharpe, rSG.null.pSharpe, rSG.nTest);
fprintf('  買進持有 Sharpe：隨機漫步 %.2f，含訊號 %.2f\n', ...
    rRW.baseline.buyHold.sharpe, rSG.baseline.buyHold.sharpe);
fprintf(['  => 隨機漫步上觀測值落在虛無分布之內，含訊號時遠在其外。\n' ...
         '     注意一：單一次結果本來就會有約 5%% 的機率偶然顯著，故不可只看\n' ...
         '     一次。框架的校準是以 40 條隨機漫步驗證 P(p<0.05) 是否接近名目\n' ...
         '     水準（見 help walkforward_backtest 的校準表），而非靠單次結果。\n' ...
         '     注意二：含訊號序列為合成資料，其 Sharpe 遠高於真實市場可得水準。\n']);

if SHOW_PLOTS
    figBt = figure('Name', 'Walk-forward backtest', 'Color', 'w', ...
        'Position', [70 70 980 700]);

    subplot(2,2,1);
    plot(tB, rRW.equity, '-', 'Color', [.2 .4 .8], 'LineWidth', 1.6);
    hold on; grid on; box on;
    plot(tB, pxRW/pxRW(1), '-', 'Color', [.6 .6 .6], 'LineWidth', 1.2);
    legend({'strategy', 'buy \& hold'}, 'Interpreter', 'latex', ...
        'Location', 'northwest', 'FontSize', 9);
    xlabel('$t$', 'Interpreter', 'latex'); ylabel('equity');
    title(sprintf('Random walk: Sharpe %.2f, $p = %.3f$', ...
        rRW.sharpe, rRW.null.pSharpe), 'Interpreter', 'latex', 'FontSize', 11);
    set(gca, 'FontSize', 9);

    subplot(2,2,2);
    plot(tB, rSG.equity, '-', 'Color', [.15 .55 .3], 'LineWidth', 1.6);
    hold on; grid on; box on;
    plot(tB, pxSG/pxSG(1), '-', 'Color', [.6 .6 .6], 'LineWidth', 1.2);
    legend({'strategy', 'buy \& hold'}, 'Interpreter', 'latex', ...
        'Location', 'northwest', 'FontSize', 9);
    xlabel('$t$', 'Interpreter', 'latex'); ylabel('equity');
    title(sprintf('With real signal: Sharpe %.2f, $p = %.3f$', ...
        rSG.sharpe, rSG.null.pSharpe), 'Interpreter', 'latex', 'FontSize', 11);
    set(gca, 'FontSize', 9);

    subplot(2,2,3);
    histogram(rRW.null.sharpe, 20, 'FaceColor', [.6 .6 .65], ...
        'EdgeColor', 'none'); hold on; grid on; box on;
    xline(rRW.sharpe, 'r-', 'LineWidth', 2);
    xlabel('Sharpe'); ylabel('count');
    title('Null distribution vs observed (random walk)', ...
        'Interpreter', 'latex', 'FontSize', 11);
    legend({'null', 'observed'}, 'Location', 'northeast', 'FontSize', 9);
    set(gca, 'FontSize', 9);

    subplot(2,2,4);
    histogram(rSG.null.sharpe, 20, 'FaceColor', [.6 .6 .65], ...
        'EdgeColor', 'none'); hold on; grid on; box on;
    xline(rSG.sharpe, 'r-', 'LineWidth', 2);
    xlabel('Sharpe'); ylabel('count');
    title('Null distribution vs observed (with signal)', ...
        'Interpreter', 'latex', 'FontSize', 11);
    legend({'null', 'observed'}, 'Location', 'north', 'FontSize', 9);
    set(gca, 'FontSize', 9);

    export_png(figBt, 'fig_backtest', EXPORT_PNG, FIG_DIR);
end


%% 11. 橫斷面多空策略 ======================================================
% 第 10 節的偵測力分析指出：單一序列在真實訊號強度下偵測不到。橫斷面透過
% (a) 樣本數增為 T x nAssets、(b) 移除當日市場成分，突破此限制。
fprintf('\n【11】橫斷面多空回測 (cross_sectional_backtest)\n');

nXO = 1200;  nXA = 100;  tX = (1:nXO).';
wSt = warning('off', 'wavelet_denoise_series:blockJump');
cleanupX = onCleanup(@() warning(wSt));

fprintf('  %-10s %-10s %-8s %-9s %-10s %s\n', ...
    'kappa', 'IC', 'ICIR', 'p(IC)', 'Sharpe', '說明');
icAll = zeros(3,1);  shAll = zeros(3,1);
kapList = [0 0.02 0.05];
for kk = 1:numel(kapList)
    pxX = local_make_panel(nXO, nXA, kapList(kk), 500 + kk);
    Fx  = wavelet_features(pxX, tX, 'Windows', [21 63]);
    rx  = cross_sectional_backtest(Fx, pxX, 'NullRuns', 100, 'CostBps', 5);
    icAll(kk) = rx.meanIC;  shAll(kk) = rx.sharpe;
    if kapList(kk) == 0
        note = '無訊號 -> 應不顯著';
    else
        note = '有訊號 -> 應顯著';
    end
    fprintf('  %-10.2f %+-10.4f %+-8.2f %-9.3f %+-10.2f %s\n', ...
        kapList(kk), rx.meanIC, rx.icir, rx.null.pIC, rx.sharpe, note);
end

% 同一份資料上，橫斷面與單一序列擇時的正面比較
pxX = local_make_panel(nXO, nXA, 0.05, 777);
Fx  = wavelet_features(pxX, tX, 'Windows', [21 63]);
rx  = cross_sectional_backtest(Fx, pxX, 'NullRuns', 100);
accS = zeros(5,1);  shS = zeros(5,1);
for j = 1:5
    Fj = wavelet_features(pxX(:,j), tX, 'Windows', [21 63]);
    rj = walkforward_backtest(Fj, pxX(:,j), 'NullRuns', 100);
    accS(j) = rj.accuracy;  shS(j) = rj.sharpe;
end
fprintf('\n  同一份資料（kappa = 0.05）：\n');
fprintf('    橫斷面多空     : IC %+.4f | Sharpe %+.2f | p(IC) %.3f\n', ...
    rx.meanIC, rx.sharpe, rx.null.pIC);
fprintf('    單一序列擇時   : 準確率 %.4f | Sharpe %+.2f （5 檔平均）\n', ...
    mean(accS), mean(shS));
fprintf('    等權買進持有   : Sharpe %+.2f\n', ...
    rx.baseline.equalWeightBuyHold.sharpe);
fprintf(['  => 同一個訊號，橫斷面偵測得到，單一序列擇時偵測不到。\n' ...
         '     注意：本節為合成資料且注入的訊號遠強於真實市場，Sharpe\n' ...
         '     數值不可作為實盤預期；其意義僅在於「框架能偵測到它該\n' ...
         '     偵測到的東西」。\n']);

if SHOW_PLOTS
    figXs = figure('Name', 'Cross-sectional', 'Color', 'w', ...
        'Position', [90 90 960 420]);
    subplot(1,2,1);
    plot(tX, rx.equity, '-', 'Color', [.15 .45 .75], 'LineWidth', 1.8);
    hold on; grid on; box on;
    plot(tX, rx.baseline.equalWeightBuyHold.equity, '-', ...
        'Color', [.6 .6 .6], 'LineWidth', 1.4);
    legend({'cross-sectional long-short', 'equal-weight buy \& hold'}, ...
        'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 10);
    xlabel('$t$', 'Interpreter', 'latex'); ylabel('equity');
    title(sprintf('Long-short equity (IC %.3f, Sharpe %.1f)', ...
        rx.meanIC, rx.sharpe), 'Interpreter', 'latex', 'FontSize', 12);
    set(gca, 'FontSize', 10);

    subplot(1,2,2);
    icv = rx.ic(isfinite(rx.ic));
    histogram(icv, 25, 'FaceColor', [.45 .6 .8], 'EdgeColor', 'none');
    hold on; grid on; box on;
    xline(0, 'k-', 'LineWidth', 1.2);
    xline(mean(icv), 'r-', 'LineWidth', 2);
    xlabel('daily rank IC'); ylabel('count');
    legend({'IC distribution', 'zero', 'mean IC'}, 'Location', 'northwest', ...
        'FontSize', 9);
    title(sprintf('Daily IC: mean %.3f, hit rate %.2f', ...
        mean(icv), mean(icv > 0)), 'Interpreter', 'latex', 'FontSize', 12);
    set(gca, 'FontSize', 10);
    export_png(figXs, 'fig_crosssection', EXPORT_PNG, FIG_DIR);
end


fprintf('\n===============================================================\n');
fprintf(' 示範結束。詳細 API 說明請執行：\n');
fprintf('   help build_chebyshev_matrices\n');
fprintf('   help wavelet_denoise_series\n');
fprintf('   help wavelet_features\n');
fprintf('   help walkforward_backtest\n');
fprintf('   help cross_sectional_backtest\n');
fprintf('===============================================================\n');


%% 局部工具函數 ===========================================================
function export_png(figHandle, name, doExport, figDir)
%EXPORT_PNG 將圖形輸出為 PNG (供 README 使用)
if ~doExport
    return;
end
if ~isfolder(figDir)
    mkdir(figDir);
end
fpath = fullfile(figDir, [name '.png']);
exportgraphics(figHandle, fpath, 'Resolution', 150, 'BackgroundColor', 'white');
fprintf('  已輸出圖檔 : %s\n', fpath);
end


function S = local_make_panel(nObs, nAssets, kappa, seed)
%LOCAL_MAKE_PANEL 合成多標的面板：市場因子 + 個股雜訊 + 可控的橫斷面反轉
%   kappa = 0 表示完全沒有橫斷面可預測性；kappa > 0 時，相對自身移動平均
%   偏離越大的標的，次期報酬越傾向反轉（即橫斷面均值回歸）。
rng(seed);
beta = 0.6 + 0.8*rand(1, nAssets);           % 各標的的市場敏感度
sigI = 0.010 + 0.008*rand(1, nAssets);       % 各標的的特異波動
rm   = 0.0003 + 0.009*randn(nObs, 1);        % 市場因子
S    = zeros(nObs, nAssets);
S(1,:) = 100;
ma     = S(1,:);
for t = 2:nObs
    dev = (S(t-1,:) - ma) ./ max(ma, eps);
    dev = dev - mean(dev);                   % 只保留橫斷面成分
    S(t,:) = S(t-1,:) .* (1 + beta.*rm(t) + sigI.*randn(1,nAssets) - kappa*dev);
    ma = 0.95*ma + 0.05*S(t,:);              % 指數移動平均
end
end


function a = local_dir_acc(X, y, splitAt)
%LOCAL_DIR_ACC 以 ridge 迴歸做方向分類，回傳樣本外準確率（依時間切分）
ok  = all(isfinite(X), 2) & isfinite(y);
idx = find(ok);
tr  = idx(idx <= splitAt);
te  = idx(idx >  splitAt);
Xtr = X(tr,:);  Xte = X(te,:);
mu  = mean(Xtr, 1);  sd = std(Xtr, 0, 1);  sd(sd == 0) = 1;
Xtr = [ones(numel(tr),1), (Xtr - mu)./sd];
Xte = [ones(numel(te),1), (Xte - mu)./sd];
w   = (Xtr.'*Xtr + 1e-3*eye(size(Xtr,2))) \ (Xtr.'*y(tr));
a   = mean(sign(Xte*w) == y(te));
end


function out = ternary(cond, a, b)
%TERNARY 三元選擇，僅供本腳本排版使用
if cond
    out = a;
else
    out = b;
end
end
