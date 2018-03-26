function If = nonLocalMeans(I, patchSize, filtSigma, patchSigma)
% NONLOCALMEANS - Non local means CPU implementation
%   
% SYNTAX
%
%   IF = NONLOCALMEANS( IN, FILTSIGMA, PATCHSIGMA )
%
% INPUT
%
%   IN          Input image                     [m-by-n]
%   PATCHSIZE   Neighborhood size in pixels     [1-by-2]
%   FILTSIGMA   Filter sigma value              [scalar]
%   PATCHSIGMA  Patch sigma value               [scalar]
%
% OUTPUT
%
%   IF          Filtered image after nlm        [m-by-n]
%
% DESCRIPTION
%
%   IF = NONLOCALMEANS( IN, PATCHSIZE, FILTSIGMA, PATCHSIGMA ) applies
%   non local means algorithm with sigma value of FILTSIGMA, using a
%   Gaussian patch of size PATCHSIZE with sigma value of PATCHSIGMA.
%
%
  
  %% USEFUL FUNCTIONS
  
  % create 3-D cube with local patches
  patchCube = @(X,w) ...
      permute( ...
          reshape( ...
              im2col( ...
                  padarray( ...
                      X, ...
                      (w-1)./2, 'symmetric'), ...
                  w, 'sliding' ), ...
              [prod(w) size(X)] ), ...
          [2 3 1] );

  % create 3D cube
  B = patchCube(I, patchSize);
  [m, n, d] = size( B );
  B = reshape(B, [ m*n d ] );
  
  % gaussian patch
  H = fspecial('gaussian',patchSize, patchSigma);
  H = H(:) ./ max(H(:));
  
  % apply gaussian patch on 3D cube
  B = bsxfun( @times, B, H' );
  
  % compute kernel
  D = squareform( pdist( B, 'euclidean' ) );
  D = exp( -D.^2 / filtSigma );
  D(1:length(D)+1:end) = max(max(D-diag(diag(D)),[],2), eps);
  
  % generate filtered image
  If = D*I(:) ./ sum(D, 2);
  
  % reshape for image
  If = reshape( If, [m n] );
  
end


%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION
%
%   0.2 - January 05, 2017
%
% CHANGELOG
%
%   0.1 (Dec 28, 2016) - Dimitris
%       * initial implementation
%   0.2 (Jan 05, 2017) - Dimitris
%       * minor fix (distance squared)
%
% ------------------------------------------------------------

