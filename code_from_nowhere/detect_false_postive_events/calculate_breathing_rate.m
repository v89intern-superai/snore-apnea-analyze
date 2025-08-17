function [f0T] = calculate_breathing_rate(flat_regionT)

flat_regionT=movmean(flat_regionT,10);
YT = fft(flat_regionT);
P2T = abs(YT/length(flat_regionT));
P1T = P2T(1:length(flat_regionT)/2+1);
P1T(2:end-1) = 2*P1T(2:end-1);
fT = 100*(0:(length(flat_regionT)/2))/length(flat_regionT);
[max_P1T,ind_max_fT]=max(P1T);
f0T=fT(ind_max_fT);
end

