
function [] = izhick_moreinput(a,b,c,d,title_graph,tau,final,type)
tspan = 0:tau:final;
if(type=="resonator")
u=-62;  w=b*u;
uu=[];  ww=[]; 
T1=tspan(end)/10;
T2=T1+20;
T3 = 0.7*tspan(end);
T4 = T3+40;
elseif(type=="integrator")
u=-60;  w=b*u;
uu=[];  ww=[]; 
T1=tspan(end)/11;
T2=T1+5;
T3 = 0.7*tspan(end);
T4 = T3+10;
elseif(type=="variability")
    uu=[];  ww=[]; 
   u=-64;  w=b*u; 
elseif(type=="bistability")
u=-61;  w=b*u;
uu=[];  ww=[]; 
T1=tspan(end)/8;
T2 = 216;
end
for t=tspan
    if(type=="resonator")
        if ((t>T1) && (t < T1+4)) || ((t>T2) && (t < T2+4)) || ((t>T3) && (t < T3+4)) || ((t>T4) && (t < T4+4)) 
        I=0.65;
        else
        I=0;
        end
    elseif(type=="integrator")
       if((t>T1) && (t < T1+2)) || ((t>T2) & (t < T2+2)) || ((t>T3) && (t < T3+2)) || ((t>T4) && (t < T4+2)) 
            I=9;
       else
            I=0;
       end
    elseif(type=="variability")
        if ((t>10) & (t < 15)) | ((t>80) & (t < 85)) 
        I=1;
        elseif (t>70) & (t < 75)
            I=-6;
        else
            I=0;
        end
    elseif(type=="bistability")
        if ((t>T1) & (t < T1+5)) | ((t>T2) & (t < T2+5)) 
            I=1.24;
        else
            I=0.24;
        end
    end
if(type=="resonator" || type=="variability" || type=="bistability")
u= u + tau*(0.04*u^2+5*u+140-w+I);
w = w + tau*a*(b*u-w);
else
u = u + tau*(0.04*u^2+4.1*u+108-w+I);
w = w+ tau*a*(b*u-w);
end
if u > 30
        uu(end+1)=30;
        u = c;
        w = w + d;
   
else
    uu(end+1)=u;
end
    ww(end+1)=w;
end
f=figure('visible','on');
subplot(2,1,1);
legend("Time","Membral Potential")
if(type=="resonator")
plot(tspan,uu,[0 T1 T1 (T1+8) (T1+8) T2 T2 (T2+8) (T2+8) T3 T3 (T3+8) (T3+8) T4 T4 (T4+8) (T4+8) max(tspan)],-90+[0 0 10 10 0 0 10 10 0 0 10 10 0 0 10 10 0 0]);
elseif(type=="integrator")
plot(tspan,uu,[0 T1 T1 (T1+2) (T1+2) T2 T2 (T2+2) (T2+2) T3 T3 (T3+2) (T3+2) T4 T4 (T4+2) (T4+2) max(tspan)],-90+[0 0 10 10 0 0 10 10 0 0 10 10 0 0 10 10 0 0]);
elseif(type=="variability")
    plot(tspan,uu,[0 10 10 15 15 70 70 75 75 80 80 85 85 max(tspan)],...
          -85+[0 0  5  5  0  0  -5 -5 0  0  5  5  0  0]);
elseif(type=="bistability")
    plot(tspan,uu,[0 T1 T1 (T1+5) (T1+5) T2 T2 (T2+5) (T2+5) max(tspan)],-90+[0 0 10 10 0 0 10 10 0 0]);

end
axis([0 max(tspan) -90 30]);
title(title_graph);
xlabel('t')
ylabel('u')
title(title_graph);

set(legend("Time","Membral Potential"),'fontsize',14)
subplot(2,1,2);
plot(uu,ww);
xlabel('membran potential')
ylabel('recovery variable')
plot_spikes(f,title_graph);
end
