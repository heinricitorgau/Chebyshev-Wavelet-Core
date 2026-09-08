function rootDir = setup_paths(varargin)
%SETUP_PATHS 將本套件的所有子目錄加入 MATLAB 搜尋路徑
%
%   本套件的程式碼分置於數個子目錄（core / pipeline / backtest / dataio /
%   demos / verify），使用前需一次將它們加入搜尋路徑。
%
%   語法：
%     setup_paths            將所有子目錄加入路徑（僅本次工作階段有效）
%     setup_paths('save')    同上，並以 savepath 永久保存
%     root = setup_paths(...)  另回傳套件根目錄的絕對路徑
%
%   路徑以本檔案的實際位置推得，因此無論從哪個工作目錄呼叫皆可正確運作。
%
%   使用範例：
%       cd('path/to/chebyshev_wavelet_core');
%       setup_paths;
%       demo_omi_pom
%
%   See also ADDPATH, SAVEPATH, PROJECT_ROOT.
%
%   Author : Kao, En-Tsai
%   License: MIT (see LICENSE)

rootDir = fileparts(mfilename('fullpath'));

subDirs = {'core', 'pipeline', 'backtest', 'dataio', 'demos', 'verify'};
added   = {};
for i = 1:numel(subDirs)
    d = fullfile(rootDir, subDirs{i});
    if isfolder(d)
        addpath(d);
        added{end+1} = subDirs{i}; %#ok<AGROW>
    else
        warning('setup_paths:missingFolder', '找不到子目錄：%s', d);
    end
end

fprintf('setup_paths: 已加入 %d 個目錄（%s）\n', numel(added), strjoin(added, ', '));

if nargin > 0 && any(strcmpi(varargin, 'save'))
    savepath;
    fprintf('setup_paths: 已以 savepath 永久保存。\n');
end

if nargout == 0
    clear rootDir;
end
end
