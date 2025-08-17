function [false_positives] = detect_false_positives_per_study(foldername)

if exist(foldername)==7
    %%read flow signals from EDF files
    path=strsplit(foldername, '\');
    patient_str=strsplit(path{1,3},'-');
    patient=str2num(patient_str{1,1});
    signal_Tflow=[];
    signal_Pflow=[];
    edf_files=dir(strcat(foldername,'\*].edf'));
    %% read flow channels from the PSG study from all availble EDF segment files
    for j=1:length(edf_files)
        filename=edf_files(j).name;
        [hdr, record] = edfread(strcat(foldername,'\',filename),'targetSignals','FlowPatient');
        record_Tflow=record(1,find(record(1,:)));
        record_Pflow=record(2,find(record(2,:)));
        signal_Tflow=[signal_Tflow; record_Tflow'];
        signal_Pflow=[signal_Pflow; record_Pflow'];
    end
    
    
    %% extract information on the annotated events by the corresponding rml
    %%files
    [family,type,start,duration,starting_date,total_duration,segment_files] = events_annotation_extraction(foldername);
    
    %% ensure only events of more than 10 s are included
    family=family(find(duration>=10));
    type=type(find(duration>=10));
    start=start(find(duration>=10));
    duration=duration(find(duration>=10));
    
    
    %% create common T/P mask by the annotated respiratory events (=apneas
    %%and other)
    k=0;
    if ~(length(family)==0) && ~isempty(find(strcmp(family,'Respiratory')))
        for i=1:length(family)
            if strcmp(family{i},'Respiratory')
                k=k+1;
                num_mask(k,1)=start(i);
                num_mask(k,2)=start(i)+duration(i);
            end
        end
        num_mask=num_mask*hdr.frequency(12);
        mask_events=ones(1,length(signal_Tflow));
        for i=1:length(num_mask)
            if num_mask(i,2)<=length(signal_Tflow)
                mask_events(num_mask(i,1):num_mask(i,2))=zeros(1,num_mask(i,2)-num_mask(i,1)+1);
            end
        end
    else 
        mask_events=ones(1,length(signal_Tflow));
    end
    
    %% detrend both signals by detecting moving average in a minimum
    %%% window of 300 samples (~= 1 respiratory cycle, fs=100Hz,
    %%% respiratory cycle=3-5s)
    
    maT=movmean(signal_Tflow,300);
    Tflowdet=signal_Tflow-maT;
    maP=movmean(signal_Pflow,300);
    Pflowdet=signal_Pflow-maP;
    
    %% extract moving rms in the same window length 
    movrms=dsp.MovingRMS(300);
    Tflow_rms=movrms(Tflowdet);
    Pflow_rms=movrms(Pflowdet);
    if mean(Tflow_rms)>1
        yupperT = envelope(Tflow_rms,150,'peak');
        Trmsstd=movstd(yupperT,300);
        mean_rms_std_T=mean(Trmsstd);
        max_rms_std_T=max(Trmsstd);
        maskstdT=Trmsstd<0.3*mean(Trmsstd);
        maskrmsT=Tflow_rms>mean(Tflow_rms)/3;
        maskT=mask_events'.*maskstdT.*maskrmsT;
        masked_T=Tflowdet.*maskT;
    end
    if mean(Pflow_rms)>1
        yupperP= envelope(Pflow_rms,150,'peak');
        Prmsstd=movstd(yupperP,300);
        maskstdP=Prmsstd<0.3*mean(Prmsstd);
        maskrmsP=Pflow_rms>mean(Pflow_rms)/3;
        maskP=mask_events'.*maskstdP.*maskrmsP;
        masked_P=Pflowdet.*maskP;
    end
    
    %% examine each event separately
    ev=1;
    k=0;
    not_ev=0;
    false_positives=[];
    while ev<=length(family)
        if duration(1,ev)>=10
            if strcmp(type{1,ev},'ObstructiveApnea')||strcmp(type{1,ev},'Hypopnea')||strcmp(type{1,ev},'MixedApnea')||strcmp(type{1,ev},'CentralApnea')
                k=k+1;
                %keep 10 s prior and after the event so that these segments
                %are used to define the flat normal breathing amplitude for
                %both signals of T:thermistor and P:pressure cannula flow
                if (start(ev)+duration(1,ev)+10)*hdr.frequency(12)<=length(signal_Tflow)
                    apnea1{k}=signal_Tflow((start(1,ev)-10)*hdr.frequency(12):(start(ev)+duration(1,ev)+10)*hdr.frequency(12));
                else
                    apnea1{k}=signal_Tflow((start(1,ev)-10)*hdr.frequency(12):end);
                end
                
                if ~length(apnea1{k})==0
                    meanapn=movmean(apnea1{k},300);
                    apnea{k}=apnea1{k}-meanapn;
                    windowSize = 50;
                    b = (1/windowSize)*ones(1,windowSize);
                    a = 1;
                    apnea{k} = filter(b,a,apnea{k});
                    if length(apnea{k}(1200:end-1200))>0
                        f0T = calculate_breathing_rate(apnea{k}(1200:end-1200));
                    else
                        length(apnea{k}(1000:end-1000))
                        f0T = calculate_breathing_rate(apnea{k}(1000:end-1000));
                    end
                    [yupper,ylower] = envelope(apnea{k},350,'peak');
                    min_ampl(k)=min(yupper(1000:end-1000)-ylower(1000:end-1000));
                    if (start(ev)+duration(1,ev)+10)*hdr.frequency(12)<=length(signal_Pflow)
                        apnea1p{k}=signal_Pflow((start(1,ev)-10)*hdr.frequency(12):(start(ev)+duration(1,ev)+10)*hdr.frequency(12));
                    else
                        apnea1p{k}=signal_Pflow((start(1,ev)-10)*hdr.frequency(12):end);
                    end
                    meanapnp=movmean(apnea1p{k},300);
                    apneap{k}=apnea1p{k}-meanapnp;
                    windowSize = 50;
                    b = (1/windowSize)*ones(1,windowSize);
                    a = 1;
                    apneap{k} = filter(b,a,apneap{k});
                    if length(apneap{k}(1200:end-1200))>0
                        f0P = calculate_breathing_rate(apneap{k}(1200:end-1200));
                    else
                        length(apneap{k}(1000:end-1000));
                        f0P = calculate_breathing_rate(apneap{k}(1000:end-1000));
                    end
                    mean_early_amplt=NaN;
                    mean_middlea_amplt=NaN;
                    mean_early_amplp=NaN;
                    mean_middlea_amplp=NaN;
                    if f0P>=0.16 && f0T>=0.16 &&(f0P<=0.40)&&(f0T<=0.40)
                        [pks,locs]=findpeaks(movmean(apnea{k},100));
                        locs=locs(pks>mean(movmean(apnea{k},100))+std(movmean(apnea{k},100))/4);
                        pks=pks(pks>mean(movmean(apnea{k},100))+std(movmean(apnea{k},100))/4);
                        [npks,nlocs]=findpeaks(-movmean(apnea{k},100));
                        nlocs=nlocs(-npks<mean(movmean(apnea{k},100))-std(movmean(apnea{k},100))/4);
                        npks=npks(-npks<mean(movmean(apnea{k},100))-std(movmean(apnea{k},100))/4);
                        [pksp,locsp]=findpeaks(movmean(apneap{k},100));
                        locsp=locsp(pksp>mean(movmean(apneap{k},100))+std(movmean(apneap{k},100))/4);
                        pksp=pksp(pksp>mean(movmean(apneap{k},100))+std(movmean(apneap{k},100))/4);
                        [npksp,nlocsp]=findpeaks(-movmean(apneap{k},100));
                        nlocsp=nlocsp(-npksp<mean(movmean(apneap{k},100))-std(movmean(apneap{k},100))/4);
                        npksp=npksp(-npksp<mean(movmean(apneap{k},100))-std(movmean(apneap{k},100))/4);
                        if ~isempty(pks(locs<1000)) && ~isempty(npks(nlocs<1000))
                            mean_early_amplt=mean(pks(locs<1000))-mean(-npks(nlocs<1000));
                        end
                        if ~isempty(pks(1500<locs & locs<length(apnea{k})-1500)) && ~isempty(npks(1500<nlocs & nlocs<length(apnea{k})-1500))
                            mean_middlea_amplt=mean(pks(1500<locs & locs<length(apnea{k})-1500))-mean(-npks(1500<nlocs & nlocs<length(apnea{k})-1500));
                        end
                        if ~isempty(pksp(locsp<1000)) && ~isempty(npksp(nlocsp<1000))
                            mean_early_amplp=mean(pksp(locsp<1000))-mean(-npksp(nlocsp<1000));
                        end
                        if ~isempty(pksp(1500<locsp & locsp<length(apneap{k})-1500)) && ~isempty(npksp(1500<nlocsp & nlocsp<length(apneap{k})-1500))
                            
                            mean_middlea_amplp=mean(pksp(1500<locsp & locsp<length(apneap{k})-1500))-mean(-npksp(1500<nlocsp & nlocsp<length(apneap{k})-1500));
                        end
                        if isnan(mean_early_amplt)||isnan(mean_middlea_amplt)||mean_early_amplt<0.7
                            mean_early_amplt=0;
                            mean_middlea_amplt=0;
                        end
                        if isnan(mean_early_amplp)||isnan(mean_middlea_amplp)||mean_early_amplp<0.25
                            mean_early_amplp=0;
                            mean_middlea_amplp=0;
                        end
                            [yupperp,ylowerp] = envelope(apneap{k},350,'peak');
                            min_ampl_p(k)=min(yupperp(1000:end-1000)-ylowerp(1000:end-1000));
                            %% detect mean_ampl_flatTandP for this specific
                            %%apnea event
                            mean_ampl_flatT(k) = max(max(yupper(500:1000)-ylower(500:1000)),max(yupper(end-1000:end-500)-ylower(end-1000:end-500)));
                            mean_ampl_flatP(k) = max(max(yupperp(500:1000)-ylowerp(500:1000)),max(yupperp(end-1000:end-500)-ylowerp(end-1000:end-500)));
                            if mean_ampl_flatT(k)<0.7
                                mean_ampl_flatT(k)=0;
                            end
                            if mean_ampl_flatP(k)<0.25
                                mean_ampl_flatP(k)=0;
                            end
                            %% 
                            if ~(mean_ampl_flatT(k)==0)&& ~(mean_ampl_flatP(k)==0)
                                if isempty(find((yupper(1000:end-1000)-ylower(1000:end-1000))<0.70*mean_ampl_flatT(k)))|| isempty(find((yupperp(1000:end-1000)-ylowerp(1000:end-1000))<0.70*mean_ampl_flatP(k)))
                                    not_ev=not_ev+1;
                                    false_positives(not_ev,:)=[patient, start(ev), start(ev)+duration(1,ev), duration(1,ev), 1*strcmp(type(ev),'ObstructiveApnea')+2*strcmp(type(ev),'Hypopnea')+3*strcmp(type(ev),'MixedApnea')+4*strcmp(type(ev),'CentralApnea')];%strcat(path(end),sprintf('%08d',ev),'.jpg');
                                    %save(strcat(foldername,'\false_positives.mat'),'false_positives');
                                end
                            else
                                if (mean_ampl_flatT(k)==0)
                                    if isempty(find((yupperp(1000:end-1000)-ylowerp(1000:end-1000))<0.70*mean_ampl_flatP(k)))
                                        not_ev=not_ev+1;
                                        false_positives(not_ev,:)=[patient, start(ev), start(ev)+duration(1,ev), duration(1,ev), 1*strcmp(type(ev),'ObstructiveApnea')+2*strcmp(type(ev),'Hypopnea')+3*strcmp(type(ev),'MixedApnea')+4*strcmp(type(ev),'CentralApnea')];
                                        %save(strcat(foldername,'\false_positives.mat'),'false_positives');
                                    end
                                end
                                if (mean_ampl_flatP(k)==0)
                                        if isempty(find((yupper(1000:end-1000)-ylower(1000:end-1000))<0.70*mean_ampl_flatT(k)))
                                            not_ev=not_ev+1;
                                            false_positives(not_ev,:)=[patient, start(ev), start(ev)+duration(1,ev), duration(1,ev), 1*strcmp(type(ev),'ObstructiveApnea')+2*strcmp(type(ev),'Hypopnea')+3*strcmp(type(ev),'MixedApnea')+4*strcmp(type(ev),'CentralApnea')];
                                            %save(strcat(foldername,'\false_positives.mat'),'false_positives');
                                        end
                                end
                            end
                    end
                end
            end
        end
        ev=ev+1;
    end
end
end

