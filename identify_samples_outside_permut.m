function testXX1f = identify_samples_outside_permut(sizeXX1f,permXX1f_downsamp)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    testXX1f = (1:sizeXX1f)';
    tmp = [];
    idx = 1;
    for ind = 1:size(testXX1f,1)
        if sum(permXX1f_downsamp==testXX1f(ind,1)) == 0
            tmp(idx,1) = testXX1f(ind,1);
            idx = idx + 1;
        end
    end
    testXX1f = tmp;
end

