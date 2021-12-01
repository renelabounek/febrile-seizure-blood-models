function output = tab_translate(input)
%TAB_TRANSLATE Summary of this function goes here
%   Detailed explanation goes here
    if ~ischar(input)
            output =input;
    else
            output = str2double(input);
    end
end

