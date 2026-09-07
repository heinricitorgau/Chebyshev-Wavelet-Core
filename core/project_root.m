function root = project_root()
%PROJECT_ROOT 回傳本套件根目錄的絕對路徑
%
%   套件內需要存取固定位置的檔案時（資料快取 data/、圖檔 figures/），
%   不可使用相對於「目前工作目錄」的路徑——否則從不同目錄執行會失敗，
%   或把檔案寫到錯誤的地方。本函數以自身所在位置推得根目錄，使這些
%   路徑與工作目錄無關。
%
%   使用範例：
%       cacheFile = fullfile(project_root(), 'data', 'fx_ecb_cache.mat');
%
%   See also SETUP_PATHS, MFILENAME.
%
%   Author : Kao, En-Tsai
%   License: MIT (see LICENSE)

% 本檔案位於 <root>/core/，故上一層即為根目錄
root = fileparts(fileparts(mfilename('fullpath')));
end
