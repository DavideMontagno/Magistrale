
function [] = izhick_neg(a,b,c,d,title_graph,tau,final,type)
tspan = 0:tau:final;
uu=[];  ww=[];
start_input_plot=-90;

if(type=="in-spiking" || type=="in-bursting")
    u=-63.8; w=b*u;
else
    u=-64;  w=b*u;
    T1=20;
end

for t=tspan
   if(type=="spike")
        if (t>T1) && (t < T1+5) 
            current=-15;
        else
            current=0;
        end
   elseif(type=="rebound")
       if (t>T1) & (t < T1+5) 
            current=-15;
       else
            current=0;
       end
   elseif(type=="in-spiking" || type=="in-bursting")
       if (t < 50) | (t>250)
        current=80;
    else
        current=75;
       end
    end
u= u + tau*(0.04*u^2+5*u+140-w+current);
w = w + tau*a*(b*u-w);
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
if(type=="spike")
    plot(tspan,uu,[0 T1 T1 (T1+5) (T1+5) max(tspan)],-85+[0 0 -5 -5 0 0]);
elseif(type=="in-spiking" || type=="in-bursting")
    plot(tspan,uu,[0 50 50 250 250 max(tspan)],-80+[0 0 -10 -10 0 0]);

else
    plot(tspan,uu,[0 T1 T1 (T1+5) (T1+5) max(tspan)],-85+[0 0 -5 -5 0 0]);

end
axis([0 max(tspan) start_input_plot 30])
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
