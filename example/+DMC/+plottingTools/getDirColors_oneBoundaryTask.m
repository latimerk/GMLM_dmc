function [cs] = getDirColors_onlyBoundaryTask(dirs)
N = numel(dirs);
cs = zeros(N,3);

taskDirection = 135;
 
for ii = 1:N
    dir = dirs(ii);
    
    while(dir < 0)
        dir = dir+360;
    end
    
    dir = mod(dir,360);

    if(dir > taskDirection-90 && dir < taskDirection+90)
        %cat 1
        cs(ii,1:2) = 0;
        cs(ii,  3) = interp1([taskDirection-90,taskDirection+90], [1.0 0.4],dir);
        
    elseif(dir > taskDirection+90 || dir < taskDirection-90)
        %cat 2
        if(dir > taskDirection+90)
            dir = dir-360;
        end
        
        cs(ii,2:3) = 0;
        cs(ii,  1) = interp1([taskDirection+90-360,taskDirection-90], [0.4 1],dir);
        
    else
        %border case: black
        cs(ii,:) = 0;
    end
end