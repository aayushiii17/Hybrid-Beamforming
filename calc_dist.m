function [dist_vec] = calc_dist(single_point,ant_pos)
% single point in the form [x, y].
% ant_pos : two-dimensional array,first column: x-coordinates second column y-coordinates of the points
%dist_vec: A column vector containing the Euclidean distances between the single point and each point in the ant_pos array
%CALC_DIST Summary of this function goes here
%   Detailed explanation goes here

num_of_pts = size(ant_pos, 1);

dist_vec = zeros(num_of_pts, 1);

for ii = 1:num_of_pts
    dist_vec(ii) = sqrt((single_point(1)-ant_pos(ii, 1))^2 + (single_point(2)-ant_pos(ii, 2))^2);
end
%Euclidean distance between the single point and that point ant_pos
end

