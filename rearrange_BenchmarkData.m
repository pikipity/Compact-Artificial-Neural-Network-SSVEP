clear;clc;close all;
data_dir='./BenchmarkData';
load(fullfile(data_dir,'Phase.mat'),'freqs','phases');
load(fullfile(data_dir,'channel_name.mat'))
fs=250;
sine_ref=[];
t0=0;
for i=1:length(freqs)
    sine_ref(i,:,:)=gen_ref_sin_tshift(freqs(i),fs,fs*6,5,phases(i),t0);
end
save('sine_ref.mat','sine_ref','-v6');

b2=[];
a2=[];
num_of_subbands=5;
for k=1:num_of_subbands
    bandpass1(1)=8*k;
    bandpass1(2)=90;
    [b2(k,:),a2(k,:)] = cheby1(4,1,[bandpass1(1)/(fs/2) bandpass1(2)/(fs/2)],'bandpass');
end

channel_select=[48 54 55 56 57 58 61 62 63];
subject_no=35;
num_of_subbands=5;
block_no=6;
start_t=0.5+0.14;
trial_no=40;

for sub_no=1:subject_no
    data_sub=[];
    for k=1:num_of_subbands
        sub=['S' num2str(sub_no)];
        disp(['Filter: ' sub ', filterbank: ' num2str(k)])
        load(fullfile(data_dir,[sub '.mat']));
        y=data;
        for trial=1:trial_no
            for block=1:block_no
                for ch=1:size(y,1)
                    temp=squeeze(y(ch,:,trial,block));
                    temp=detrend(temp);
                    y(ch,:,trial,block)=filtfilt(b2(k,:),a2(k,:),temp);
                end
                if floor(start_t*fs)<1
                    data_sub(k,trial,block,:,:)=squeeze(y(channel_select,1:end,trial,block));
                else
                    data_sub(k,trial,block,:,:)=squeeze(y(channel_select,floor(start_t*fs):end,trial,block));
                end
            end
        end
    end
    save(['sub_' num2str(sub_no) '.mat'],'data_sub','-v6');
end
    