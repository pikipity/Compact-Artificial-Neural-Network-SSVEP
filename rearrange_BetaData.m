clear;clc;close all;
data_dir='./BetaData';
load(fullfile(data_dir,'Phase.mat'))
load(fullfile(data_dir,'channel_name.mat'))

fs=250;
t0=0.5+0.13;

sine_ref=[];
for i=1:length(freqs)
    sine_ref(i,:,:)=gen_ref_sin(freqs(i), fs, fs*6, 5, phases(i));
end
save('sine_ref.mat','sine_ref','-v6');

subject_no=70;
num_of_subbands=5;
block_no=4;
start_t=0.5;
trial_no=40;

f0=50;
q=35;
bw=(f0/(fs/2))/q;
[notch_b,notch_a]=iircomb(fs/f0,bw,'notch');

b2=[];
a2=[];
for k=1:num_of_subbands
    bandpass1(1)=8*k;
    bandpass1(2)=90;
    [b2(k,:), a2(k,:)] = cheby1(4,1,[bandpass1(1)/(fs/2) bandpass1(2)/(fs/2)], 'bandpass');
end

T=2+0.13;
for sub_no=1:subject_no
    
    sub=['S' num2str(sub_no)];
    store_file = ['sub_' num2str(sub_no) '_allch.mat'];
    
    load(fullfile(data_dir, [sub '.mat']))
    data = data.EEG;
    y=data;
    data = [];
    
    for k=1:num_of_subbands
        for trial=1:trial_no
            
            disp(['Process ' sub ', subband' num2str(k) ', f' num2str(trial)])
            
            for block=1:block_no
                for ch=1:size(y,1)
                    temp=squeeze(y(ch,floor(start_t*fs):floor((start_t+T)*fs-1),block,trial));
                    temp=detrend(temp);
                    temp=filtfilt(notch_b,notch_a,temp);
                    temp=filtfilt(b2(k,:), a2(k,:), temp);
                    temp=detrend(temp);
                    data(k,trial,block,ch,:)=temp;
                end
            end
        end
    end
    
    if sum(isnan(data))>0
        error('Error')
    end
    
    disp(['Save ' sub])
    save(store_file,'data','-v6')
end