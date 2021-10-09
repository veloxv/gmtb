function [ p, corrected ] = gmpd_correct_result( p, correction_type)
%GMPD_CORRECT_RESULT corrects some errors in median computation (values < 0, sum = 1)

    corrected = false;

    if (correction_type == 0)
        return;
    end
    
    
    if (any(p < 0) || any(p > 1) || abs(sum(p)-1) > 10*eps)
        if (correction_type == 1)
            p = p - min(p(:));
        else
            p(p<0) = 0;
        end
        
        p = p / sum(p);
        
        corrected = true;
        
        %warning('Result was corrected')
    end


end

