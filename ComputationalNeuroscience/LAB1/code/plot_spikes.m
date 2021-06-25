function [] = plot_spikes(figure,title)
str = pwd;
path = strcat(str,"/figures/");
savefig(strcat(path,strcat(title,".fig")));
end

