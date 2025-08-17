function [all_deleted_apneas] = extract_all_not_apnea_events()
%%% This function extracts all false positive detected events of apnea or
%%% hypopnea from all patients in the dataset. Prerequisits are the folders
%%% of stored PSG studies where corresponding EDF files of each specific
%%% PSG study is stored. The structure of the output variable
%%% "all_deleted_apneas" is an array in which each line corresponds to
%%% different detected event. Each line comprises 6 columns: (a) the PSG
%%% study number, (b) the start time of the apnea/hypopnea detected as false
%%% positive, (c) the end time of the apnea/hypopnea detected as false
%%% positive, (d) the duration of the apnea/hypopnea detected as false
%%% positive and (e) the type of the apnea/hypopnea detected as false
%%% positive according to the following:

%%% 1: corresponds to Obstructive apnea
%%% 2: corresponds to hypopnea
%%% 3: corresponds to mixed apnea
%%% 4: corresponds to central apnea


% define the folder of PSG studies
apnea_folder='F:\APNEA';
listing = dir(apnea_folder);

for i=1:length(listing)
    if ~strcmp(listing(i).name,'.') && ~strcmp(listing(i).name,'..')
        foldername{i}=strcat(listing(i).folder,'\',listing(i).name);
        % for each specific PSG study call the function to detect the false
        % positive events
        [false_postives] = find_not_apnea_events_simple(foldername{i});
        all_deleted_apneas=[all_deleted_apneas; false_postives];
        save('all_deleted_apneas.mat','all_deleted_apneas');
        fclose all;
    end
end
end

