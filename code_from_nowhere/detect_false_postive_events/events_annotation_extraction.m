function [family,type,start,duration,starting_date,total_duration,segment_files] = events_annotation_extraction(foldername)
%%% This function reads the rml files of each specific PSG study defined in
%%% foldername and returns the array of events annotated in this study by
%%% giving the family of each event, its type, its starting point and its
%%% total duration. Additionally it reads the segment files expected to be
%%% included in this study, that is the number of different EDF files to
%%% read so that the entire night is examined.

path=regexp(foldername,'\', 'split');
filename=strcat(foldername,'\',path(end),'.rml');
filename=filename{1,1};
% filenamexlsx=strcat(foldername,'\events.xlsx');
fileID=fopen(filename);
tline = fgetl(fileID);
i=1;
while ischar(tline)
    strlines{i}=tline;
    i=i+1;
    tline = fgetl(fileID);
end
for i=1:length(strlines)
    strlinesv{i}=strsplit(strlines{i});
end

f=1;
t=1;
s=1;
d=1;
segment_files=0;
for i=1:length(strlinesv)
    temp=strlinesv{1,i};
    if size(temp,2)>1
        if strcmp(temp{1,2},'<Event')
            familyv=strsplit(temp{1,3},'=');
            familyv=strsplit(familyv{1,2},'"');
            family{f}=familyv{1,2};
            f=f+1;
            typev=strsplit(temp{1,4},'=');
            typev=strsplit(typev{1,2},'"');
            type{t}=typev{1,2};
            t=t+1;
            startv=strsplit(temp{1,5},'=');
            startv=strsplit(startv{1,2},'"');
            start(s)=str2double(startv{1,2});
            s=s+1;
            durationv=strsplit(temp{1,6},'=');
            durationv=strsplit(durationv{1,2},'"');
            duration(d)=str2double(durationv{1,2});
            d=d+1;
        else
            if strcmp(temp{1,2},'<Session>')
                starting_time_line=strlinesv{1,i+1};
                temp_start_time=strsplit(starting_time_line{2},{'<','T','>'});
                starting_date=temp_start_time{1,3};
                total_duration_time=strlinesv{1,i+2};
                temp_total_duration=strsplit(total_duration_time{2},{'<Duration>','</Duration>'});
                total_duration=str2num(temp_total_duration{1,2});
            else
                if strcmp(temp{1,2},'<Segment>')
                    segment_files=segment_files+1;
                end
            end
        end
    end
end
end

