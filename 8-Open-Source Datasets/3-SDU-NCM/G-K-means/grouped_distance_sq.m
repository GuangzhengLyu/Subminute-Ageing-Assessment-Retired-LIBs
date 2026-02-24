function dist2 = grouped_distance_sq(X, C, groupDims, w)
% grouped_distance_sq - 计算分组加权欧氏距离平方
% dist2(i,k) = sum_g w(g) * || X_i(g) - C_k(g) ||^2
%
% 输入：
%   X: N×D
%   C: K×D
%   groupDims: [1 2 6 4]
%   w: 组权重（长度=组数）
% 输出：
%   dist2: N×K

[N, D] = size(X);
K = size(C,1);
assert(size(C,2) == D, '中心维度必须与X一致。');
assert(sum(groupDims) == D, 'groupDims之和必须等于D。');
assert(numel(w) == numel(groupDims), 'w长度必须等于组数。');

dist2 = zeros(N, K);

idx = [0, cumsum(groupDims)];
for g = 1:numel(groupDims)
    cols = (idx(g)+1):idx(g+1);

    Xg = X(:, cols);      % N×d_g
    Cg = C(:, cols);      % K×d_g

    % 计算欧氏距离平方：||x-c||^2 = sum((x-c).^2)
    % 利用展开： (x^2) + (c^2) - 2*x*c'
    X2 = sum(Xg.^2, 2);                 % N×1
    C2 = sum(Cg.^2, 2)';                % 1×K
    cross = Xg * Cg';                   % N×K
    dg2 = X2 + C2 - 2*cross;            % N×K

    dist2 = dist2 + w(g) * dg2;
end
end
