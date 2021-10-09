function [dist_func, grad_func, grad_func_no_emb, sod_func, epsilon, beta, prevent_zero ] = gmpd_select_distance_functions( distance )
%GMPD_SELECT_DISTANCE_FUNCTIONS returns all necessary distance functions and parameters

    switch distance
        
        case 'mean'
            dist_func = @(p1,p2) sum((p1-p2).^2);
            grad_func = [];
            grad_func_no_emb = [];
            sod_func = @(alpha,set_p,weights) NaN;
            epsilon = 0;
            beta = 0;
            prevent_zero = false;
        
        case 'hellinger'
            dist_func = @gmpd_hellinger_dist;
            grad_func = @gmpd_hellinger_grad;
            grad_func_no_emb = @gmpd_hellinger_grad_no_emb;
            sod_func = @gmpd_hellinger_sod;
            epsilon = 0.0001;
            beta = 0.8;
            prevent_zero = false;
        
        case 'bhattaccharyya'
            dist_func = @gmpd_bhattacharyya_dist;
            grad_func = @gmpd_bhattacharyya_grad;
            grad_func_no_emb = @gmpd_bhattacharyya_grad_no_emb;
            sod_func = @gmpd_bhattacharyya_sod;            
            epsilon = 0.0001;
            beta = 0.8;
            prevent_zero = false;
            
        case 'total-variation'
            dist_func = @gmpd_total_variation_dist;
            grad_func = @gmpd_total_variation_grad;
            grad_func_no_emb = @gmpd_total_variation_grad_no_emb;
            sod_func = @gmpd_total_variation_sod;
            epsilon = 0.0001; 
            beta = 0.8;
            prevent_zero = false;
            
        case 'kullback-leibler1'
            dist_func = @gmpd_kullback_leibler1_dist;
            grad_func = @gmpd_kullback_leibler1_grad;
            grad_func_no_emb = @gmpd_kullback_leibler1_grad_no_emb;
            sod_func = @gmpd_kullback_leibler1_sod;
            epsilon = 0.0001;
            beta = 0.8; 
            prevent_zero = true;
            
        case 'kullback-leibler2'
            dist_func = @gmpd_kullback_leibler2_dist;
            grad_func = @gmpd_kullback_leibler2_grad;
            grad_func_no_emb = @gmpd_kullback_leibler2_grad_no_emb;
            sod_func = @gmpd_kullback_leibler2_sod;
            epsilon = 0.0001;
            beta = 0.8; 
            prevent_zero = true;
            
        case 'jensen-shannon'
            dist_func = @gmpd_jensen_shannon_dist;
            grad_func = @gmpd_jensen_shannon_grad;
            grad_func_no_emb = @gmpd_jensen_shannon_grad_no_emb;
            sod_func = @gmpd_jensen_shannon_sod;
            epsilon = 0.0001;
            beta = 0.8; 
            prevent_zero = true;
            
        case 'lukaszyk-karmowski'
            dist_func = @gmpd_lukaszyk_karmowski_dist;
            grad_func = @gmpd_lukaszyk_karmowski_grad;
            grad_func_no_emb = @gmpd_lukaszyk_karmowski_grad_no_emb;
            sod_func = @gmpd_lukaszyk_karmowski_sod;   
            epsilon = 0.0001;
            beta = 0.8;
            prevent_zero = false;
            
        
        case 'uniform'
        % the gradient function is wrong, and no good gradient is possible
            dist_func = @gmpd_uniform_dist;
            grad_func = @gmpd_uniform_grad;
            grad_func_no_emb = @gmpd_uniform_grad_no_emb;
            sod_func = @gmpd_uniform_sod;   
            epsilon = 0.0001;
            beta = 0.8;
            prevent_zero = false;
            
            
        case 'separation1'
         %the gradient is wrong, and no good gradient possible (at least not easily)
            dist_func = @gmpd_separation1_dist;
            grad_func = @gmpd_separation1_grad;
            grad_func_no_emb = @gmpd_separation1_grad_no_emb;
            sod_func = @gmpd_separation1_sod;   
            epsilon = 0.0001;
            beta = 0.8;
            prevent_zero = false;
            
            
        case 'chi-squared1'
            dist_func = @gmpd_chi_squared1_dist;
            grad_func = @gmpd_chi_squared1_grad;
            grad_func_no_emb = @gmpd_chi_squared1_grad_no_emb;
            sod_func = @gmpd_chi_squared1_sod;
            epsilon = 0.0001;
            beta = 0.8;
            prevent_zero = false;
            
        case 'chi-squared2'
            dist_func = @gmpd_chi_squared2_dist;
            grad_func = @gmpd_chi_squared2_grad;
            grad_func_no_emb = @gmpd_chi_squared2_grad_no_emb;
            sod_func = @gmpd_chi_squared2_sod;
            epsilon = 0.0001; 
            beta = 0.8;
            prevent_zero = false;
         

        otherwise
            error('unknown distance');
                
    end

end

