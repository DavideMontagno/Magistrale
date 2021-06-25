


%tonic-spiking
izhick_sameinput(0.02,0.2,-65,6, 'Tonic-spiking- Input: 14',14,0.25,300);
%phasic-spiking
izhick_sameinput(0.02,0.25,-65,6, 'Phasic-spiking- Input: 1',1,0.1,100);
%tonic-bursting
izhick_sameinput(0.02,0.2,-50,2, 'Tonic-bursting- Input: 14',14,0.25,300);
%phasic-bursting - VEDERE
izhick_sameinput(0.02,0.25,-55,0.05, 'Phasic-bursting- Input: 1',1,0.25,300);
%Mixed-mode
izhick_sameinput(0.02,0.2,-55,4, 'Mixed-mode - Input: 14',14,0.25,300);
%Spike Frequency Adaptation
izhick_sameinput(0.01,0.2,-65,8, 'Spike Frequency Adaptation - Input: 14',14,0.25,300);
%Class 1
izhick_varinput(0.02,-0.1,-55,6, 'Class 1 Excitable',0.25,300,"exc1");
%Class 2
izhick_varinput(0.2,0.26,-65,0, 'Class 2 Excitable',0.25,300,"exc2");
%Spike Latency
izhick_oneinput(0.02,0.2,-65,6,'Spike Latency - Input: 18',18,0.1,15,1,"normal");
%Subthreshold Oscillations
izhick_oneinput(0.05,0.26,-60,0,'Subthreshold Oscillations',18,0.25,200,2,"oscillation");
%Resonator
izhick_moreinput(0.1,0.26,-60,-1,'Resonator',0.1,400,"resonator");
%Integrator
izhick_moreinput(0.02,-0.1,-55,6,'Integrator',0.25,100,"integrator");
%Rebound Spike
izhick_neg(0.03,0.25,-60,4,'Rebound Spike',0.2,200,'spike');
%Rebound burst
izhick_neg(0.03,0.25,-52,0,'Rebound burst',0.2,200,'rebound');
%Threshold variability
izhick_moreinput(0.03,0.25,-60,4,'Threshold Variability',0.25,100,"variability");
%Bistability
izhick_moreinput(0.1,0.26,-60,0,'Bistability',0.25,300,"bistability");
%Depolarizing
izhick_oneinput(1,0.2,-60,-21,"Depolarizing after-potential",0,0.1,50,0,"depolarizing")
%Accomodation
izhick_varinput(0.02,1,-55,4, 'Accomodation',0.5,400,"accomodation");
%Inhibition-induced spiking
izhick_neg(-0.02,-1,-60,8,'Inhibition-induced spiking',0.5,350,'in-spiking');
%Inhibition-induced bursting
izhick_neg(-0.026,-1,-45,-2,'Inhibition-induced bursting',0.5,350,'in-bursting');
