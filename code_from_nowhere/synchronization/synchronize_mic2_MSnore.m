function [Dfinal] = synchronize_mic2_MSnore(patient_number)
%%% this function is used to synchronize the tracheal audio recordings
%%% saved in .WAV files with the tracheal low-quality audio recordings
%%% acquired via the PSG system and stored in channel 11 of the original
%%% EDF files. The synchronization requires the patient_number of the
%%% specific study to synchronize and provides the delay between the two
%%% signals in s (Dfinal). An excel file of the correspondence between the
%%% specific study and the wav file names is required to be provided by the
%%% health care personnel supervising the study and extracting edf files
%%% from the Sleepware G3 software. 

%% Define the folder where .WAV files are stored
recordings_folder='F:\APNEA_RECORDINGS\';
listing = dir(recordings_folder);
for i=1:length(listing)
    recname{i}=strsplit(listing(i).name,'_');
end
c=[];
for n=1:size(recname,2)
    if size(recname{n},2)>1
        c(n)=str2num(recname{n}{2});
    end
end
str_signal_files={};

foldername=strcat('F:\APNEA\', sprintf( '%08d', patient_number ), '-100507');
path=regexp(foldername,'\', 'split');
if exist(foldername)==7
    segment_files=length(dir(fullfile(foldername,'\*].edf'))); %find the total number of segment files for the specific study
    signal=[];
    for j=1:segment_files
        filename=strcat(foldername,'\', path(end),sprintf('[00%d].edf',j));
        if exist(filename{1,1},'file')==2
            [hdr, record] = edfread(filename{1,1},'targetSignals','Snore');%read the MSnore signal of the PSG study stored as the 11th signal of the original EDF files
            signal=[signal; record'];
        end
    end
    s1=signal(10*60*hdr.frequency(11):(20*60)*hdr.frequency(11))/max(abs(signal(10*60*hdr.frequency(11):(20*60)*hdr.frequency(11)))); %keep only 10 s of the signal at the beginning of the audio recording
    
    %% read the excel file where the numbering of the acquired wav files that correspond to this specific study are given for all studies
    filenamexlsx_nos = strcat(recordings_folder,'Potirakis(6).xls');
    file_line=52;
    acq_number=xlsread(filenamexlsx_nos,1,strcat('B',num2str(file_line)));
    while acq_number <= patient_number
        if (acq_number == patient_number)
            first_rec_file = xlsread(filenamexlsx_nos,1,strcat('F',num2str(file_line)));
            index_recname_a = find(c-first_rec_file==0);
            if ~size(index_recname_a)==[1,1]
                index_recname = index_recname_a(1,1);
            else
                index_recname = index_recname_a;
            end
            first_rec_file_name=strcat(recname{index_recname}{1},'_',recname{index_recname}{2},'_',recname{index_recname}{3});
            [s2,freq]= audioread(strcat(recordings_folder, first_rec_file_name),[10*60*48000, (20*60)*48000]); %% reads first related file, only 10 min
            s2=s2(:,2);
            s2=s2/max(abs(s2));
            acq_number=patient_number+1;
        else
            file_line=file_line+1;
            acq_number=xlsread(filenamexlsx_nos,1,strcat('B',num2str(file_line)));
        end
    end
    
    %% synchronize the two signal after filtering and downsamping at the lowest sampling frequency (500 Hz)
    if ~isempty(s1)&&~isempty(s2)
        dt1=1/hdr.frequency(11);
        t1 = 0:dt1:(length(s1)*dt1)-dt1;
        dt2=1/freq;
        t2 = 0:dt2:(length(s2)*dt2)-dt2;
        % Filter the signal
        fc = 250; % Make higher to hear higher frequencies.
        % Design a Butterworth filter.
        [b, a] = butter(6,fc/(freq/2));
        % Apply the Butterworth filter.
        s2_filtered = filter(b, a, s2);
        s2_resam=s2_filtered(1:dt1*freq:length(s2_filtered));
        %extract the signal envelope
        [yupper1,ylower1] = envelope(s1,2000,'peak');
        %figure(1);plot(t1,s1);hold on;plot(t1,yupper1);hold on;plot(t1,ylower1);hold off;
        [yupper2,ylower2] = envelope(s2_resam,2000,'peak');
        %figure(2);plot(t1,s2_resam);hold on;plot(t1,yupper2);hold on;plot(t1,ylower2);hold off;
        D(1) = finddelay(yupper1,yupper2);
        D(2) = finddelay(ylower1,ylower2);
        D(3) = finddelay(yupper1,-ylower2);
        D(4) = finddelay(-ylower1,yupper2);
        temp_delay(:)=D(:)/500;
        for d=1:4
            additional_part=zeros(round(abs(temp_delay(d)))*500,1);
            if temp_delay(d)>0
                s1_final=[additional_part; s1];
                s2_resam_final=s2_resam;
            else 
                s2_resam_final=[additional_part; s2_resam];
                s1_final=s1;
            end
            totlen=min(length(s1_final),length(s2_resam_final));
            error_estimator(d)=mean(abs((s1_final(1:totlen))/max(abs(s1_final(1:totlen)))-(s2_resam_final(1:totlen)/max(abs(s2_resam_final(1:totlen))))));
        end
        [min_error,min_err_index]=min(error_estimator);
        Dfinal=temp_delay(min_err_index);
        if ~isempty(Dfinal)
            additional_part=zeros(round(abs(Dfinal*500)),1);
            dt1=1/hdr.frequency(11);
            t1 = 0:dt1:(length(s1)*dt1)-dt1;
            dt2=1/freq;
            t2 = 0:dt2:(length(s2)*dt2)-dt2;
            if Dfinal>0
                s1_final=[additional_part; s1];
                t1=t1+Dfinal;
            else 
                s2_resam=[additional_part; s2_resam];
                s1_final=s1;
                t2=t2+Dfinal;
            end
            %%figures providing visual ispection of the result
            figure; subplot(2,1,1);plot(1:length(s1_final),s1_final);subplot(2,1,2);plot(1:length(s2_resam),s2_resam);
            figure; subplot(2,1,1);plot(t1,s1);subplot(2,1,2);plot(t2,s2);
        end
    end
end
end

