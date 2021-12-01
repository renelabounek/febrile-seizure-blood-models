function clr = color_decision(p,thr)
%COLOR_DECISION Summary of this function goes here
%   Detailed explanation goes here
    if p< thr
            clr = [0.2 0.7 0.2];
    elseif p<0.05
            clr = [0 0 0];
    else
            clr=[0.6 0.6 0.6];
    end
end

