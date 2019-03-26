function [ d_res, z_res, Dz, obj_val ] = admm_learn(b, kernel_size,...
    lambda_residual, lambda_prior, ...
    max_it, tol, ...
    verbose, init, smooth_init)
% ref: "Restoration of Poissonian Images Using Alternating Direction
% Optimization" and " FRAME-BASED IMAGE DEBLURRING WITH UNKNOWN BOUNDARY CONDITIONS
% USING THE ALTERNATING DIRECTION METHOD OF MULTIPLIERS".


%Kernel size contains kernel_size = [psf_s, psf_s, k]
psf_s = kernel_size(1); % spatial size of the filters.
k = kernel_size(end); % number of learned filters.
%     n = size(b,5);
n = size(b,4); % number of samples.

%PSF estimation
psf_radius = floor( psf_s/2 );
%      size_x = [size(b,1) + 2*psf_radius, size(b,2) + 2*psf_radius, size(b,3), size(b,4), n]; %
size_x = [size(b,1) + 2*psf_radius, size(b,2) + 2*psf_radius, size(b,3), n]; % dataset size after spatial convolution.
size_z = [size_x(1), size_x(2), k, n]; % 2D feature maps of dimension size_x(1)*size_x(2)
%     size_k_full = [size_x(1), size_x(2), size_x(3), size_x(4), k]; % [spatial_x, spatial_y, spectral, num of examples, number of filters]
size_k_full = [size_x(1), size_x(2), size_x(3), k];
%     size_zhat = [size_x(1), size_x(2), 1, 1, k, n]; % spectrum of feature maps, each 3D hyperspectral cube has a single 2D feature map.
size_zhat = [size_x(1), size_x(2), k, n];

%Smooth offset
%     smoothinit = padarray( smooth_init, [psf_radius, psf_radius, 0, 0, 0], 'symmetric', 'both');% zero padded the low frequency component of data.
smoothinit = padarray( smooth_init, [psf_radius, psf_radius, 0, 0], 'symmetric', 'both');% zero padded the low frequency component of data.

% Objective
objective = @(z, dh) objectiveFunction( z, dh, b, lambda_residual, lambda_prior, psf_radius, size_z, size_x, smoothinit );

%Prox for masked data
[M, Mtb] = precompute_MProx(b, psf_radius, smoothinit); 
%对应到公式里面M就是MtM因为M是一个单位阵，M和MtM是相等的, spatially padded mask; Mtb是掩码和规范化后的数据is the masked and
%normalized data.相当于公式中的Mtx
ProxDataMasked = @(u, theta) (Mtb + 1/theta * u ) ./ ( M + 1/theta * ones(size_x) ); % adding 1/theta * u to the offset term Mtb, and cancling the mask.
%这个函数的输入为Dz求得的值也是Dz，通过这样的表达来在每一次迭代的过程中来最小化Dz
%Prox for sparsity
ProxSparse = @(u, theta) max( 0, 1 - theta./ abs(u) ) .* u;
%这个函数的输入为Zk，输出为最小化的Zk。

%Prox for kernel constraints
ProxKernelConstraint = @(u) KernelConstraintProj( u, size_k_full, psf_radius);
%这个函数的输入为Zk输出为Zk。

%% Pack lambdas and find algorithm params
% lambda = [lambda_residual, lambda_prior]; % weights for data fedelity and spasity items respectively.
% gamma_heuristic = 60 * lambda_prior * 1/max(b(:));% gamma_heuristic =  lambda_prior * 1/max(b(:));% 
% gammas_D = [gamma_heuristic / 5000, gamma_heuristic];% penalty factors for {Dz, z} respectively.
% gammas_Z = [gamma_heuristic, gamma_heuristic];% gammas_Z = [gamma_heuristic / 500, gamma_heuristic];% penalty factors for {Zd, d} respectively.

lambda = [lambda_residual, lambda_prior]; % weights for data fedelity and spasity items respectively.
gamma_heuristic = 60 * lambda_prior * 1/max(b(:));% gamma_heuristic =  lambda_prior * 1/max(b(:));% 
gammas_D = [gamma_heuristic, gamma_heuristic];% gammas_D = [gamma_heuristic, gamma_heuristic];% gammas_D = [gamma_heuristic / 5000, gamma_heuristic];% penalty factors for {Dz, z} respectively.
gammas_Z = [gamma_heuristic, gamma_heuristic];% gammas_Z = [gamma_heuristic, gamma_heuristic];%
%% Initialize variables for K. in the following variables, ...
%%the first item is for the mask data fidelity (Zd), and the second
%%item is for the filter constraint (d). d represents the filters.
varsize_D = {size_x, size_k_full}; % data set Zd size and filter set size.
xi_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) }; % intermediate variable (u_D+d_D) for the quadratic problem of filters d.
xi_D_hat = { zeros(varsize_D{1}), zeros(varsize_D{2}) };

u_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) }; % slack variables in terms of primary variable (filters d): {u_D{1}=Zd, u_D{2}=d}.
d_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) }; % dual variables.
v_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) }; % intermediate variables for storing the values of { Zd, d}

%Initial iterates
if ~isempty(init)
    d_hat = init.d;
    d = [];
else % pad filters to the full convolution dimensions and circularly shift to account for the boundary effect.
    %         d = padarray( randn(kernel_size([1 2 5])), [size_x(1) - kernel_size(1), size_x(2) - kernel_size(2), 0], 0, 'post');%
    d = padarray( randn(kernel_size([1 2 3])), [size_x(1) - kernel_size(1), size_x(2) - kernel_size(2), 0], 0, 'post');% inititialize filters to random noise
    d = circshift(d, -[psf_radius, psf_radius, 0] );
    %         d = permute(repmat(d, [1 1 1 kernel_size(3) kernel_size(4)]), [1 2 4 5 3]);
    d = repmat(d, [1 1 1 k]); % (x,y,spectral, filter)进行平铺到filer个数，
    d_hat = fft2(d);
end

%% Initialize variables for Z. in the following variables, ...
%%the first item is for the mask data fidelity (Dz), and the second
%%item is for the sparse map constraint (z). z represents the coeff maps.
%     varsize_Z = {size_x([1 2 3 4 5]), size_z};
varsize_Z = {size_x, size_z};
xi_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
xi_Z_hat = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };

u_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
d_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) }; % dual variables.
v_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };% intermediate variables for storing the values of { Dz, z}

z = randn(size_z);

%Initial vals
obj_val = objective(z, d_hat);

if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
    fprintf('Iter %d, Obj %3.3g, Diff %5.5g\n', 0, obj_val, 0)
end

%Iteration for local back and forth
max_it_d = 10; % max_it_d = 10;% 
max_it_z = 10; % max_it_z = 10; %

obj_val_filter = obj_val;
obj_val_z = obj_val;

%Iterate
for i = 1:max_it
    %% Update kernels
%     %Timing
%     tic;
    
    %Recompute what is necessary for kernel convterm later
    rho = gammas_D(2)/gammas_D(1);
    obj_val_min = min(obj_val_filter, obj_val_z);
    d_old = d;
    d_hat_old = d_hat;
    
%     %Timing
%     t_kernel = toc;

    z_hat = fft2(z); %reshape( fft2(reshape(z, size_z(1),size_z(2),size_z(3),[])), size_zhat );
    %size_zhat=106x106x20x5
    [zhatT_blocks, invzhatTzhat_blocks] = myprecompute_Z_hat_d(z_hat, gammas_D);
    %返回了需要的矩阵逆 和 特殊格式的z hat
    %这是一种套路通过声明三个变量  来
    for i_d = 1:max_it_d
        
%         %Timing
%         tic;
        
        %Compute v_i = H_i * z.  fake slack variables that are computed from the primary variables (filters d).
        %             v_D{1} = real(ifft2( reshape(sum( bsxfun(@times, d_hat, permute(z_hat, [1 2 5 6 3 4])), 5), size_x) ));
        v_D{1} = real(ifft2( reshape(sum( bsxfun(@times, d_hat, permute(z_hat, [1 2 5 3 4])), 4), size_x) ));% d_hat : (x,y,spectral,filters),
        % z_hat : (x,y,filters,examples)  ,  permute(z_hat, [1 2 5 3 4]) : (x,y,spectral(1),filters,,examples).
        %permute用来重新排列
        %这里算出来还正好就是图像b的大小。
        v_D{2} = d;
        
        %Compute proximal updates. true slack variables that need to be updated by the solution of proximal operators.
        u_D{1} = ProxDataMasked( v_D{1} - d_D{1}, lambda(1)/gammas_D(1) );%输入为Dz- Dz的running errors，输出的更新的Dz
        u_D{2} = ProxKernelConstraint( v_D{2} - d_D{2});%输入为d- d的残差 就是lambda
        
        for c = 1:2
            %Update running errors. dual variables.这个的意思是说Dz的running
            %errors和 对偶变量lambda
            d_D{c} = d_D{c} - (v_D{c} - u_D{c});
            %计算对偶变量
            %Compute new xi and transform to fft. construct the
            %variables used for solving the primary variable ( filters
            %d).
            xi_D{c} = u_D{c} + d_D{c}; % xi_D{1} has the same size with Zd. xi_D{2} has the same size with d.
            xi_D_hat{c} = fft2(xi_D{c});
            %那么说xi_D就是Dz和d
        end
        
        %Solve convolutional inverse. the filters' solution by a
        %quadratic peroblem.
        %         d_hat = solve_conv_term_D(z_hat, xi_D_hat, rho, size_z);
        d_hat = mysolve_conv_term_D(zhatT_blocks, invzhatTzhat_blocks, xi_D_hat, gammas_D, size_k_full, n);
        d = real(ifft2( d_hat ));
        
%         %Timing
%         t_kernel_tmp = toc;
%         t_kernel = t_kernel + t_kernel_tmp;
        
        obj_val = objective(z, d_hat);
        if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
            fprintf('--> Obj %5.5g \n', obj_val )
        end
    end
    
    %         obj_val_filter_old = obj_val_old;
    obj_val_filter = obj_val;
    
    %Debug progress 
    d_diff = d - d_old;
    d_comp = d;
    if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
        obj_val = objective(z, d_hat);
        fprintf('Iter D %d, Obj %5.5g, Diff %5.5g\n', i, obj_val, norm(d_diff(:),2)/ norm(d_comp(:),2))
    end
    
    %% Update sparsity term
    
%     %Timing
%     tic;
    
    %Recompute what is necessary for convterm later
    %     [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(d_hat, size_x);% D'D.Once the
    %     dhatT_flat = conj(permute(dhat_flat, [3 2 1])); %Same for all images
    
    %     [dhatT_blocks, dhatTdhat_blocks] = myprecompute_H_hat_Z(d_hat, gammas_Z);% for each spatial frequency index i, compute the inv[dTd_(i)_rho*I] and store into invdhatTdhat_blocks; same with dhatT_blocks.
    [dhatT_blocks, invdhatTdhat_blocks] = myprecompute_H_hat_Z(d_hat, gammas_Z);
    
    
    z_hat = fft2(z);
    z_old = z;
    z_hat_old = z_hat;
    
%     %Timing
%     t_vars = toc;
    
    for i_z = 1:max_it_z
        
%         %Timing
%         tic;
        
        %Compute v_i = H_i * z
        %             v_Z{1} = real(ifft2(squeeze(sum(bsxfun(@times, d_hat, permute(z_hat, [1 2 5 6 3 4])), 5))));
        v_Z{1} = real(ifft2(squeeze(sum(bsxfun(@times, d_hat, permute(z_hat, [1 2 5 3 4])), 4))));% Dz
        v_Z{2} = z;
        
        %Compute proximal updates
        u_Z{1} = ProxDataMasked( v_Z{1} - d_Z{1}, lambda(1)/gammas_Z(1) );
        u_Z{2} = ProxSparse( v_Z{2} - d_Z{2}, lambda(2)/gammas_Z(2) );
        
        for c = 1:2
            %Update running errors
            d_Z{c} = d_Z{c} - (v_Z{c} - u_Z{c});
            
            %Compute new xi and transform to fft
            xi_Z{c} = u_Z{c} + d_Z{c};
            xi_Z_hat{c} = fft2( xi_Z{c} );
        end
        
        %Solve convolutional inverse
        %             z_hat = solve_conv_term_Z(dhatT_flat, dhatTdhat_flat, xi_Z_hat, gammas_Z, size_z, kernel_size(3)*kernel_size(4));
        %         z_hat = solve_conv_term_Z(dhatT_flat, dhatTdhat_flat, xi_Z_hat, gammas_Z, size_z, kernel_size(3));
        %         z_hat = mysolve_conv_term_Z(dhatT_blocks, dhatTdhat_blocks, xi_Z_hat, gammas_Z, size_z, kernel_size(3));
        z_hat = mysolve_conv_term_Z(dhatT_blocks, invdhatTdhat_blocks, xi_Z_hat, gammas_Z, size_z, kernel_size(3));
        
        z = real(ifft2(z_hat));
        
%         %Timing
%         t_vars_tmp = toc;
%         t_vars = t_vars + t_vars_tmp;
        
        obj_val = objective(z, d_hat);
        if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
            fprintf('--> Obj %5.5g \n', obj_val )
        end
        
    end
    
    obj_val_z = obj_val;
    
    %         if obj_val_min <= obj_val_filter && obj_val_min <= obj_val_z
    %             z_hat = z_hat_old;
    %             z = reshape( real(ifft2( reshape(z_hat, size_x(1), size_x(2), []) )), size_z );
    %
    %             d_hat = d_hat_old;
    %             d = real(ifft2( d_hat ));
    %
    %             obj_val = objective(z, d_hat);
    %             break;
    %         end
    
    %Debug progress
    z_diff = z - z_old;
    z_comp = z;
    if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
        fprintf('Iter Z %d, Obj %5.5g, Diff %5.5g, Sparsity %5.5g\n', i, obj_val, norm(z_diff(:),2)/ norm(z_comp(:),2), nnz(z(:))/numel(z(:)))
    end
    
    %Termination
    if norm(z_diff(:),2)/ norm(z_comp(:),2) < tol && norm(d_diff(:),2)/ norm(d_comp(:),2) < tol
        break;
    end
end

%Final estimate
z_res = z;

d_res = circshift( d, [psf_radius, psf_radius, 0, 0] );
d_res = d_res(1:psf_radius*2+1, 1:psf_radius*2+1, :, :);

z_hat = reshape(fft2(z), size_zhat);
Dz = real(ifft2(squeeze(sum(bsxfun(@times, d_hat, permute(z_hat, [1 2 5 3 4])), 4)))) + smoothinit;% Dz
% Dz = real(ifft2( reshape(sum(bsxfun(@times, d_hat, z_hat), 5), size_x) )) + smoothinit;

return;

function [u_proj] = KernelConstraintProj( u, size_k_full, psf_radius)

%Get support
%u就是dk
u_proj = circshift( u, [psf_radius, psf_radius, 0, 0] );
%将d分裂进行循环移位回来
u_proj = u_proj(1:psf_radius*2+1, 1:psf_radius*2+1, :, :, :);
%然后d取11x11的大小，通过这里应该是传入的u是很大的，然后取11x11的大小，
%Normalize
u_norm = repmat( sum(sum(u_proj.^2, 1),2), [size(u_proj,1), size(u_proj,2), 1, 1] );
u_proj( u_norm >= 1 ) = u_proj( u_norm >= 1 ) ./ sqrt(u_norm( u_norm >= 1 ));

%Now shift back and pad again 补上然后移位变回去。从这里可以看出输出还是d
u_proj = padarray( u_proj, (size_k_full - size(u_proj)), 0, 'post');
u_proj = circshift(u_proj, -[psf_radius, psf_radius, 0, 0] );

return;

% eliminate the boundary effect--only measure data fedality within original
% image boundary.消除边界效应，只测量原始数据的长度。
function [M, Mtb] = precompute_MProx(b, psf_radius,smoothinit)

M = padarray(ones(size(b)), [psf_radius, psf_radius, 0, 0]);% the mask M is all through that is 0-padded to the convolutional size.
Mtb = padarray(b, [psf_radius, psf_radius, 0, 0]).*M - smoothinit.*M;% pad b and mask ,then remove the low-frequency component.

return;

% arrange the filters' spectra into 3D which keeps the two spatial
% dimensions and collaps all other dimensions into the third dimension.
function [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(dhat, size_x )
% Computes the spectra for the inversion of all H_i

%Precompute the dot products for each frequency
dhat_flat = reshape( dhat, size_x(1) * size_x(2), size_x(3)*size_x(4), [] );
dhatTdhat_flat = sum(conj(dhat_flat).*dhat_flat,3);

return;


% for each spatial frequency index i, compute the inv[dTd_(i)_rho*I] and store into invdhatTdhat_blocks; same with dhatT_blocks.
function [zhatT_blocks, invzhatTzhat_blocks] = myprecompute_Z_hat_d(z_hat, gammas)
%计算后返回 某种形式的z_hat 和计算好的（ZtZ+rho*I）-1
%size_zhat=106x106x20x5
sy = size(z_hat,1); sx = size(z_hat,2); k = size(z_hat,3); n = size(z_hat,4);

invzhatTzhat_blocks = zeros(k,k,sx * sy);%20,20,106x106,这里没写错
rho = gammas(2)/gammas(1);
%为什么要转换为这种形式？Z是有一种特殊的形式的
zhatT_blocks = conj( permute( reshape(z_hat, sx * sy, k, n), [2,3,1]) ); %  permute( reshape(z_hat, sx * sy, k, n), [2,3,1]) rearranges z_hat into dimension k*n*(sx*sy).
%这里将z_hat reshape为k，n，sx*sy，conj计算共轭值。输出的维度和输入的维度一致。
%然后对于每一个像素进行处理，计算的是矩阵的逆。
for i = 1:sx*sy
    invzhatTzhat_blocks(:,:,i) = pinv(rho * eye(k) + zhatT_blocks(:,:,i) * zhatT_blocks(:,:,i)');% inv[zTz_(i)+rho*I]
end
%这样invzhatTzhat_blocks的大小为20x20x106x106
%zhatT_blocks的格式为k,n,sx*sy 20 5 106*106；

return;


% for each spatial frequency index i, compute the inv[dTd_(i)_rho*I] and store into invdhatTdhat_blocks; same with dhatT_blocks.
function [dhatT_blocks, invdhatTdhat_blocks] = myprecompute_H_hat_Z(d_hat, gammas)
%这里就是返回了
sy = size(d_hat,1); sx = size(d_hat,2); sw = size(d_hat,3); k = size(d_hat,4); 
invdhatTdhat_blocks = zeros(k,k,sx * sy);
rho = gammas(2)/gammas(1);

dhatT_blocks = conj( permute( reshape(d_hat, sx * sy, sw, k), [3,2,1]) ); %  permute( reshape(d_hat, sx * sy, sw, k), [2,1,3]) rearranges d_hat into dimension sw*k*(sx*sy).

for i = 1:sx*sy
    invdhatTdhat_blocks(:,:,i) = pinv(rho * eye(k) + dhatT_blocks(:,:,i) * dhatT_blocks(:,:,i)');% inv[dTd_(i)_rho*I]
end

return;


%
function d_hat = mysolve_conv_term_D(zhatT_blocks, invzhatTzhat_blocks, xi_hat, gammas, size_d, n)

% Solves sum_j gamma_i/2 * || H_j d - xi_j ||_2^2
% In our case: 1/2|| Zd - xi_1 ||_2^2 + rho * 1/2 * || d - xi_2||
% with rho = gamma(2)/gamma(1)

%Size size_d为106x106x31x20
%     size_Dz为106x106x31x5,所以说最后算出来的是20x20
sy = size_d(1); sx = size_d(2); sw = size_d(3); k = size_d(4);

d_hat = zeros(k,sw,sx*sy);%20 31 106x106
x1_blocks = permute(reshape(xi_hat{1}, sy*sx, sw, n), [3,2,1]);% samples x wavelengths x spatial frequencies.x1 are synthesized images Zd
x2_blocks = permute( reshape(xi_hat{2}, sy*sx, sw, k), [3,2,1] );% filters x wavelengths x spatial frequencies, x2 are filters d
%Dz:106x106 31 5 reshape 5 31   d:106x106 31 20 reshape 20 31
%Rho
rho = gammas(2)/gammas(1); %rho = sw * gammas(2)/gammas(1);

%Compute filters in blocks of each spatial frequency, d_hat now is filters x wavelengths x spatial frequencies
for i=1:sx*sy
    %通过对每一个元素进行计算，
    d_hat(:,:,i) = invzhatTzhat_blocks(:,:,i) * ( zhatT_blocks(:,:,i) * x1_blocks(:,:,i) + rho * x2_blocks(:,:,i) );                                                                
end

%Final transpose gives d_hat in (sy,sx,sw,k)
d_hat = reshape(permute(d_hat, [3,2,1]), size_d);

return;


%Rho
%rho = gammas(2)/gammas(1);
function d_hat = solve_conv_term_D(z_hat, xi_hat, rho, size_z )

% Solves sum_j gamma_i/2 * || H_j d - xi_j ||_2^2
% In our case: 1/2|| Zd - xi_1 ||_2^2 + rho * 1/2 * || d - xi_2||
% with rho = gamma(2)/gamma(1)

%Size
%     sy = size_z(1); sx = size_z(2); sw = size(xi_hat{1},3)*size(xi_hat{1},4); k = size_z(3); n = size_z(4);
sy = size_z(1); sx = size_z(2); sw = size(xi_hat{1},3); k = size_z(3); n = size_z(4);

%Reshape to cell per frequency
xi_hat_1_cell = num2cell( permute( reshape(xi_hat{1}, sx * sy * sw, n), [2,1] ), 1);% each cell in xi_hat_1_cell (n-by-1) contains all synthetized images' FFT values with the same spatial frequency indices.
xi_hat_2_cell = num2cell( permute( reshape(xi_hat{2}, sx * sy * sw, k), [2,1] ), 1);% each cell in xi_hat_2_cell  (k-by-1) contains all estimated filters' FFT values with the same spatial frequency indices.
zhat_mat = reshape( num2cell( permute( reshape(z_hat, [sy*sx, k, n] ), [3,2,1] ), [1 2] ), [1 sy*sx]); %each cell (n-by-k) in zhat_mat contains all examples' all filters' FFT values with the same spatial frequency indices.

%Invert
x = cell(size(xi_hat_1_cell));% used to store the FFT estimates of filters, each cell corresponds to a spatial frequency.
for i=1:sx*sy % per spatial frequency. efficiently invert Z'Z+rho*I using the matrix inversion lemma and variable reordering.
    opt = (1/rho * eye(k) - 1/rho * zhat_mat{i}'*pinv(rho * eye(n) + zhat_mat{i}*zhat_mat{i}')*zhat_mat{i});% since the coeff maps of all wavelength are assumed identical, the inversed submatrix is same for all wavelengths.
    for j=1:sw % per wavelength
        ind = (j-1)*sx*sy + i; % jump over a wavelength band.
        x{ind} = opt*(zhat_mat{i}' * xi_hat_1_cell{ind} + rho * xi_hat_2_cell{ind});
    end
end

%Reshape to get back the new Dhat
d_hat = reshape( permute(cell2mat(x), [2,1]), size(xi_hat{2}) );% cell2mat(x) is (k,sx * sy * sw)

return;

function z_hat = solve_conv_term_Z(dhatT, dhatTdhat, xi_hat, gammas, size_z, sw )


% Solves sum_j gamma_i/2 * || H_j z - xi_j ||_2^2
% In our case: 1/2|| Dz - xi_1 ||_2^2 + rho * 1/2 * || z - xi_2||
% with rho = gamma(2)/gamma(1)
sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);

%Rho
rho = sw * gammas(2)/gammas(1);

%Compute b
b = squeeze(sum(bsxfun(@times, dhatT, permute(reshape(xi_hat{1}, sy*sx, sw, n), [4,2,1,3])),2)) + rho .* permute( reshape(xi_hat{2}, sy*sx, k, n), [2,1,3] );

%Invert
scInverse = ones([1,sx*sy]) ./ ( rho * ones([1,sx*sy]) + sum(dhatTdhat,2).');

x = 1/rho*b - 1/rho * bsxfun(@times, bsxfun(@times, scInverse, squeeze(sum(dhatTdhat,2))'), b);

%Final transpose gives z_hat
z_hat = reshape(permute(x, [2,1,3]), size_z);

return;


function z_hat = mysolve_conv_term_Z(dhatT_blocks, invdhatTdhat_blocks, xi_hat, gammas, size_z, sw )


% Solves sum_j gamma_i/2 * || H_j z - xi_j ||_2^2
% In our case: 1/2|| Dz - xi_1 ||_2^2 + rho * 1/2 * || z - xi_2||
% with rho = gamma(2)/gamma(1)
sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);
z_hat = zeros(k,n,sx*sy);
x1_blocks = permute(reshape(xi_hat{1}, sy*sx, sw, n), [2,3,1]);% wavelengths x samples x spatial frequencies.x1 are synthesized images
x2_blocks = permute( reshape(xi_hat{2}, sy*sx, k, n), [2,3,1] );% filters x samples x spatial frequencies, x2 are coeffs

%Rho
rho = gammas(2)/gammas(1); %rho = sw * gammas(2)/gammas(1);

%Compute coeffs in blocks of each spatial frequency, z_hat now is filters x samples x spatial frequencies
for i=1:sx*sy
    z_hat(:,:,i) = invdhatTdhat_blocks(:,:,i) * ( dhatT_blocks(:,:,i) * x1_blocks(:,:,i) + rho * x2_blocks(:,:,i) );
end

%Final transpose gives z_hat in (sy,sx,k,n)
z_hat = reshape(permute(z_hat, [3,1,2]), size_z);

return;

% function z_hat = mysolve_conv_term_Z(dhatT_blocks, dhatTdhat_blocks, xi_hat, gammas, size_z, sw )
%
%
% % Solves sum_j gamma_i/2 * || H_j z - xi_j ||_2^2
% % In our case: 1/2|| Dz - xi_1 ||_2^2 + rho * 1/2 * || z - xi_2||
% % with rho = gamma(2)/gamma(1)
% sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);
% z_hat = zeros(k,n,sx*sy);
% x1_blocks = permute(reshape(xi_hat{1}, sy*sx, sw, n), [2,3,1]);% wavelengths x samples x spatial frequencies.x1 are synthesized images
% x2_blocks = permute( reshape(xi_hat{2}, sy*sx, k, n), [2,3,1] );% filters x samples x spatial frequencies, x2 are coeffs
%
% %Rho
% rho = gammas(2)/gammas(1); % rho = sw * gammas(2)/gammas(1);%
%
% %Compute coeffs in blocks of each spatial frequency, z_hat now is filters x samples x spatial frequencies
% for i=1:sx*sy
%     z_hat(:,:,i) = dhatTdhat_blocks(:,:,i) \ ( dhatT_blocks(:,:,i) * x1_blocks(:,:,i) + rho * x2_blocks(:,:,i) );
% end
%
% %Final transpose gives z_hat in (sy,sx,k,n)
% z_hat = reshape(permute(z_hat, [3,1,2]), size_z);
%
% return

% % for each spatial frequency index i, compute the [dTd_(i)_rho*I] and store into dhatTdhat_blocks; same with dhatT_blocks.
% function [dhatT_blocks, dhatTdhat_blocks] = myprecompute_H_hat_Z(d_hat, gammas)
%
% sy = size(d_hat,1); sx = size(d_hat,2); sw = size(d_hat,3); k = size(d_hat,4);
% dhatTdhat_blocks = zeros(k,k,sx * sy);
% rho = gammas(2)/gammas(1);
%
% dhatT_blocks = conj( permute( reshape(d_hat, sx * sy, sw, k), [3,2,1]) ); %  permute( reshape(d_hat, sx * sy, sw, k), [2,1,3]) rearranges d_hat into dimension sw*k*(sx*sy).
%
% for i = 1:sx*sy
%     dhatTdhat_blocks(:,:,i) = rho * eye(k) + dhatT_blocks(:,:,i) * dhatT_blocks(:,:,i)';% [dTd_(i)_rho*I]
% end
% return

function f_val = objectiveFunction( z, d_hat, b, lambda_residual, lambda, psf_radius, size_z, size_x, smoothinit)

%Params
%     n = size_z(4);

%Data term and regularizer
%     z2 = permute(repmat(z, [1 1 1 1 size_x(3)]), [1 2 5 3 4]);
%     zhat = reshape( fft2(reshape(z2,size_z(1),size_z(2),size_z(3),[])), size(z2) );
%     Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,1,n]) .* zhat, 4),
%     size_x) )) + smoothinit; % (x,y,spectral,filter,example)

%     z_hat = permute(fft2(z), [1 2 5 6 3 4]);% (x,y,1,1,filter,example)
z_hat = permute(fft2(z), [1 2 5 3 4]);% (x,y,1,filter,example)
%     Dz = real(ifft2( squeeze(sum(bsxfun(@times, d_hat, z_hat),5)) )) + smoothinit;
Dz = real(ifft2( squeeze(sum(bsxfun(@times, d_hat, z_hat),4)) )) + smoothinit;% replicate the spectral dimension in z_hat and the
% example dimension in d_hat. d_hat .* z_ha = (x,y,spectral,filter,example).
%Whenever a dimension of A or B is singleton (equal to one), bsxfun virtually replicates
%the array along that dimension to match the other array. In the case where a dimension of A or B is singleton,
%and the corresponding dimension in the other array is zero, bsxfun virtually diminishes the singleton dimension to zero.

%     f_z = lambda_residual * 1/2 * norm( reshape( Dz(1 + psf_radius:end - psf_radius, ...
%             1 + psf_radius:end - psf_radius,:,:,:) - b, [], 1) , 2 )^2;
f_z = lambda_residual * 1/2 * norm( reshape( Dz(1 + psf_radius:end - psf_radius, ...
    1 + psf_radius:end - psf_radius,:,:) - b, [], 1) , 2 )^2;
g_z = lambda * sum( abs( z(:) ), 1 );

%Function val
f_val = f_z + g_z;

return;
