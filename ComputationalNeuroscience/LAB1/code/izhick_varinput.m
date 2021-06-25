

function [] = izhick_varinput(a,b,c,d,title_graph,tau,final,type)
 tspan = 0:tau:final;
 
uu=[];  ww=[];
current_time=[];
input=[];
start_input_plot=-100;

if(type=="exc1")
u=-60;  w=b*u; T1=30;
elseif(type=="exc2")
    u=-64;  w=b*u; T1=30;
elseif(type=="accomodation")
    u=-65; w=-16; II=[];
end
for t=tspan
    if(type=="accomodation")
         if (t < 200)
        current=t/25;
        elseif t < 300
            current=0;
        elseif t < 312.5
            current=(t-300)/12.5*4;
        else
            current=0;
          end
    else
        if(type=="exc1")
            if(t>T1)
            current=(0.075*(t-T1));
            else
                current=0;
            end
        elseif(type=="exc2")
             if(t>T1)
            current=-0.5+(0.015*(t-T1));
             else
                 current=-0.5;
            end
        end
    end
if(type=="normal" || type=="exc2")
u= u + tau*(0.04*u^2+5*u+140-w+current);
w = w + tau*a*(b*u-w);
elseif(type=="exc1")
u= u + tau*(0.04*u^2+4.1*u+108-w+current);
w = w + tau*a*(b*u-w);
else
 u = u + tau*(0.04*u^2+5*u+140-w+current);
 w = w + tau*a*(b*(u+65));
end
if u > 30
        uu(end+1)=30;
        u = c;
        w = w + d;
   
else
    uu(end+1)=u;
end
    ww(end+1)=w;
     if(type=="accomodation")
          II(end+1)=current;
     else
         
        current_time(end+1)=t;
        input(end+1)=current;
     end
end
f=figure('visible','on');
subplot(2,1,1);
legend("Time","Membral Potential")
if(type=="normal" || type=="exc1" || type=="exc2")
input = input/5;
plot(tspan,uu,current_time,start_input_plot+input);
else
  plot(tspan,uu,tspan,II*1.5-90);
end
axis([0 max(tspan) start_input_plot 30])
xlabel('t')
ylabel('u')
title(title_graph);
subplot(2,1,2);

set(legend("Time","Membral Potential"),'fontsize',14)
plot(uu,ww);
xlabel('membran potential')
ylabel('recovery variable')
plot_spikes(f,title_graph);

end
