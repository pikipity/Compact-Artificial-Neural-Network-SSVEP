function y=gen_ref_sin_tshift(f,fs,L,N,phase,t0)
    t=linspace(0,(L-1)/fs,L);
    y=[];
    for n=1:N
        y=[y;...
            sin(2.*pi.*n.*f.*(t-t0)+n.*phase);
            cos(2.*pi.*n.*f.*(t-t0)+n.*phase)];
    end
end