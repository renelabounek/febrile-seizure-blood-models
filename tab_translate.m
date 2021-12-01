function output = tab_translate(input)
%TAB_TRANSLATE Summary of this function goes here
%   Detailed explanation goes here
%
%   HELP IN PROGRESS
%
% =====================================================================================================================================================================
% Please, cite as:
% 
% Papez Jan, Labounek Rene, Jabandziev Petr, Ceska Katarina, Slaba Katerina, Oslejskova Hana, Aulicka Stefania, Nestrasil Igor. Predictive multivariate linear
% mixture models of febrile seizure risk and recurrency: a prospective case-control study. Developmental Medicine & Child Neurology (2022) [Under Review] 
%
% =====================================================================================================================================================================
% Copyright 2020-2022 Rene Labounek (1,*), Igor Nestrasil (1)
%
% (1) Division of clinical Behavioral Neuroscience, Department of Pediatrics, Masonic Institute for the Developing Brain, University of Minnesota, Minneapolis, MN, USA
% (*) email: rlaboune@umn.edu
%
% febrile-seizure-blood-models is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or any later version.
%
% febrile-seizure-blood-models is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
% or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License along with 
% febrile-seizure-blood-models. If not, see https://www.gnu.org/licenses/.
% =====================================================================================================================================================================
    if ~ischar(input)
            output =input;
    else
            output = str2double(input);
    end
end

