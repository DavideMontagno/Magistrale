
function [] = izhick_sameinput(a,b,c,d,title_graph,volt,tau,final)
 tspan = 0:tau:final;
u=-70;  w=b*u;
uu=[];  ww=[];
step = 5;
current_time=[];
input=[];
display_input=10;
start_input_plot=-90;
for t=tspan
    if(t<step)
      current=0;
      
    else 
       current=volt;
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
plot(tspan,uu,[0 step step final],-90+[0 0 10 10]);
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
