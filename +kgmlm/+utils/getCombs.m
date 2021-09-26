%alternate to combvec which more flexible argument and so that I don't need the deep learning toolbox
function [combs] = getCombs(mm)

combs = ones(numel(mm), prod(mm));
for ii = 2:size(combs,2)
    for jj = 1:numel(mm)
        combs(jj, ii) = mod(combs(jj, ii-1), mm(jj)) + 1;
        if(combs(jj, ii) ~= 1)
            combs(jj+1:end, ii) = combs(jj+1:end, ii-1);
            break;
        end
    end
end


end